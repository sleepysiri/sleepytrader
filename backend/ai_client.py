"""
AI client — batch trading decisions via Groq / Together / OpenRouter.
50 agents per API call, decisions returned as JSON array.
"""
import os
import json
import re
from typing import Optional, List, Dict, Any

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


SYSTEM_PROMPT = (
    "You are running a realistic stock market simulation. "
    "You will receive current market data and a list of trader profiles. "
    "For EACH trader, decide what they do this round based on their unique personality. "
    "Traders with opposing personalities must make opposing decisions when signals are mixed. "
    "CRITICAL: Reply with ONLY a valid JSON array — no markdown, no code block, no explanation."
)


class AIClient:
    def __init__(self):
        self.groq_key      = os.environ.get("GROQ_API_KEY", "")
        self.together_key  = os.environ.get("TOGETHER_API_KEY", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._session: Optional[Any] = None

    def has_api_key(self) -> bool:
        return bool(self.groq_key or self.together_key or self.openrouter_key)

    async def _get_session(self):
        if not AIOHTTP_AVAILABLE:
            return None
        if self._session is None or self._session.closed:
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    # ── Public batch method ───────────────────────────────────────────────────
    async def batch_decisions(
        self,
        agents_batch: List[dict],
        market_data: Dict[str, Any],
    ) -> List[dict]:
        """
        Send one prompt containing `agents_batch` profiles + market context to the LLM.
        Returns a list of {id, action, quantity, thought} dicts.
        Falls back to hold for every agent if the API is unavailable.
        """
        if not self.has_api_key() or not AIOHTTP_AVAILABLE:
            return _hold_all(agents_batch, "No API key configured.")

        user_prompt = _build_user_prompt(agents_batch, market_data)
        raw = await self._call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=2_800)

        if not raw:
            return _hold_all(agents_batch, "API call failed.")

        parsed = _parse_json(raw)
        if not parsed:
            print(f"[AIClient] JSON parse failed. Raw snippet: {raw[:300]}")
            return _hold_all(agents_batch, "JSON parse failed.")

        # Index by id, fill missing with hold
        by_id = {int(d.get("id", -1)): d for d in parsed if isinstance(d, dict)}
        result = []
        for a in agents_batch:
            d = by_id.get(a["id"])
            if d:
                result.append({
                    "id":       int(d.get("id", a["id"])),
                    "action":   str(d.get("action", "hold")).lower(),
                    "quantity": max(0, int(d.get("quantity", 0))),
                    "thought":  str(d.get("thought", "..."))[:200],
                })
            else:
                result.append({"id": a["id"], "action": "hold", "quantity": 0, "thought": "No decision returned."})
        return result

    # ── LLM router ────────────────────────────────────────────────────────────
    async def _call_llm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if self.groq_key:
            r = await self._call_groq(system, user, max_tokens)
            if r:
                return r
        if self.together_key:
            r = await self._call_together(system, user, max_tokens)
            if r:
                return r
        if self.openrouter_key:
            r = await self._call_openrouter(system, user, max_tokens)
            if r:
                return r
        return None

    async def _call_groq(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            session = await self._get_session()
            if not session:
                return None
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens":  max_tokens,
                    "temperature": 0.75,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                text = await resp.text()
                print(f"[Groq] HTTP {resp.status}: {text[:200]}")
        except Exception as e:
            print(f"[Groq] {e}")
        return None

    async def _call_together(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            session = await self._get_session()
            if not session:
                return None
            async with session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.together_key}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens":  max_tokens,
                    "temperature": 0.75,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Together] {e}")
        return None

    async def _call_openrouter(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            session = await self._get_session()
            if not session:
                return None
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[OpenRouter] {e}")
        return None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_user_prompt(agents_batch: List[dict], market_data: Dict[str, Any]) -> str:
    price      = market_data.get("price", 100.0)
    change_pct = market_data.get("change_pct", 0.0)
    rsi        = market_data.get("rsi", 50.0)
    macd_hist  = market_data.get("macd_hist", 0.0)
    recent     = market_data.get("recent_prices", [price])[-6:]
    bb_pos     = market_data.get("bb_position", "middle")   # upper/middle/lower
    sentiment  = ("BULLISH" if change_pct > 1.5
                  else "BEARISH" if change_pct < -1.5
                  else "NEUTRAL")

    market_str = (
        f"MARKET: price=${price:.2f}, change={change_pct:+.2f}%, "
        f"RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}, "
        f"BB_position={bb_pos}, sentiment={sentiment}, "
        f"recent_prices={[round(p, 2) for p in recent]}"
    )

    agents_json = json.dumps(agents_batch, separators=(",", ":"))

    return (
        f"{market_str}\n\n"
        f"TRADERS (p=personality, cash=USD, sh=shares held, prev=last action):\n"
        f"{agents_json}\n\n"
        "Rules: buy(qty*price<=cash), sell(qty<=sh), hold(qty=0). "
        "thought=1 sentence in-character.\n"
        'Return ONLY JSON: [{"id":N,"action":"buy"|"sell"|"hold","quantity":N,"thought":"..."}]'
    )


def _parse_json(text: str) -> Optional[List[dict]]:
    """Robustly extract a JSON array from LLM output."""
    text = text.strip()

    # 1. Direct parse
    try:
        d = json.loads(text)
        if isinstance(d, list):
            return d
    except Exception:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        d = json.loads(cleaned)
        if isinstance(d, list):
            return d
    except Exception:
        pass

    # 3. Regex extract first [...] block
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, list):
                return d
        except Exception:
            pass

    # 4. Fix trailing commas and single quotes
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    fixed = fixed.replace("'", '"')
    try:
        d = json.loads(fixed)
        if isinstance(d, list):
            return d
    except Exception:
        pass

    return None


def _hold_all(agents_batch: List[dict], reason: str) -> List[dict]:
    return [{"id": a["id"], "action": "hold", "quantity": 0, "thought": reason}
            for a in agents_batch]

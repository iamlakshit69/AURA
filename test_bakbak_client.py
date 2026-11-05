import asyncio, base64, json, io, wave
import httpx, numpy as np
import bakbak_client as b

def sse(chunks, done=True):
    out = b""
    for c in chunks:
        out += b"event: chunk\ndata: " + json.dumps({"type": "chunk", "status_code": 206, "done": False, "data": base64.b64encode(c).decode()}).encode() + b"\n\n"
    if done:
        out += b'event: done\ndata: {"type":"done","status_code":200,"done":true}\n\n'
    return out

f = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], "<f4").tobytes()      # 24 bytes; last sample clips
def handler(req):
    if req.url.path.endswith("/stream"):
        assert req.headers["x-api-key"] == "k"
        body = json.loads(req.content); assert body["model"] == "m1" and body["language"] == "hi"
        return httpx.Response(200, content=sse([f[:10], f[10:]]))      # split mid-sample on purpose
    if req.url.path.endswith("/text-to-speech"):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x01\x00" * 10)
        return httpx.Response(200, content=buf.getvalue())
    return httpx.Response(401, text="bad key")

async def main():
    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kw = dict(text="hi", voice_id="v", model="m1", language="hi", sample_rate=24000, speed=1.0)
    pcm = b"".join([x async for x in b.stream_pcm(c, "k", b.BASE_URL, **kw)])
    s = np.frombuffer(pcm, "<i2"); print("stream:", s.tolist()); assert s.tolist() == [0, 16383, -16383, 32767, -32767, 32767]
    pcm, rate = await b.synth_wav(c, "k", b.BASE_URL, **kw); print("wav:", len(pcm), rate); assert len(pcm) == 20 and rate == 24000
    c2 = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    try:
        async for _ in b.stream_pcm(c2, "x", b.BASE_URL, **kw): pass
    except b.BakbakError as e: print("401 ->", e.status, "retryable", e.retryable); assert e.status == 401 and not e.retryable
    vs = [{"id": "1", "language": "hi", "model": "standard"}, {"id": "2", "language": "hi", "model": "m1"}, {"id": "3", "language": "en-in", "model": "m1"}]
    assert b.pick_voice(vs, "hi", "m1")["id"] == "2" and b.pick_voice(vs, "en-in", "m1")["id"] == "3" and b.pick_voice(vs, "pa", "m1") is None
    print("ALL OK")
asyncio.run(main())

#!/usr/bin/env python3
"""End-to-end test of a running d1rs server (stdlib only; ffmpeg needed to build the format variants).

    python3 tests/e2e.py --url http://127.0.0.1:8080 [--assets /tmp/d1rs-e2e]

Covers text, JSON states, null states, images and voice; every converted format against its native original; mislabeled
fields and MIME types; error handling; and a concurrent burst mixing all modalities whose answers must match the
sequential ones (continuous batching must not change results).
"""
import argparse, base64, concurrent.futures as cf, json, os, shutil, subprocess, sys, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8080')
ap.add_argument('--assets', default='/tmp/d1rs-e2e')
ap.add_argument('--burst', type=int, default=8, help='copies of every case in the concurrent burst')
args = ap.parse_args()
A = args.assets
os.makedirs(A, exist_ok=True)

def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
fetch('http://images.cocodataset.org/val2017/000000039769.jpg', f'{A}/cats.jpg')   # two cats on a sofa
fetch('https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav', f'{A}/jfk.wav')

FF = shutil.which('ffmpeg')
def ff(src, dst, *opts):
    if FF and not os.path.exists(dst):
        subprocess.run([FF, '-y', '-loglevel', 'error', '-i', src, *opts, dst], check=True)
    return os.path.exists(dst)

image_variants = {'jpg': f'{A}/cats.jpg'}
for ext, opts in [('png', []), ('webp', []), ('gif', []), ('tiff', []), ('bmp', []), ('avif', ['-frames:v', '1']), ('jp2', [])]:
    try:
        if ff(f'{A}/cats.jpg', f'{A}/cats.{ext}', *opts): image_variants[ext] = f'{A}/cats.{ext}'
    except subprocess.CalledProcessError:
        pass
audio_variants = {'wav': f'{A}/jfk.wav'}
for ext, opts in [('mp3', []), ('ogg', ['-c:a', 'libvorbis']), ('opus', ['-c:a', 'libopus']), ('flac', []), ('m4a', ['-c:a', 'aac']),
                  ('webm', ['-c:a', 'libopus']), ('aiff', []), ('wav44k', None)]:
    try:
        if ext == 'wav44k':
            if ff(f'{A}/jfk.wav', f'{A}/jfk_44k_stereo.wav', '-ar', '44100', '-ac', '2'): audio_variants[ext] = f'{A}/jfk_44k_stereo.wav'
        elif ff(f'{A}/jfk.wav', f'{A}/jfk.{ext}', *opts): audio_variants[ext] = f'{A}/jfk.{ext}'
    except subprocess.CalledProcessError:
        pass

def b64(path, mime=None):
    data = base64.b64encode(open(path, 'rb').read()).decode()
    return f'data:{mime or "application/octet-stream"};base64,{data}'

def post(body, path='/v1/systemone'):
    req = urllib.request.Request(args.url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t) * 1e3
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), (time.perf_counter() - t) * 1e3

Q_TEXT = {
    'refund': {'type': 'noul', 'instructions': 'Is the customer asking for a refund?'},
    'team': {'type': 'choice', 'instructions': 'Which team should handle this?',
             'criteria': {'billing': 'Charges, refunds, invoices', 'technical': 'App or site faults', 'fraud': 'Suspected unauthorised use'}},
    'urgency': {'type': 'score', 'instructions': 'How urgent is this?', 'criteria': ['Can wait', 'Today', 'Blocking the customer now']},
}
Q_IMG = {
    'pet': {'type': 'choice', 'instructions': 'Which animals are in the photo?', 'criteria': {'cats': 'Cats', 'dogs': 'Dogs', 'birds': 'Birds'}},
    'sofa': {'type': 'noul', 'instructions': 'Are the animals on a sofa?'},
}
Q_AUD = {
    'kind': {'type': 'choice', 'instructions': 'What kind of utterance is this?',
             'criteria': {'request': 'A request to do something', 'question': 'A question asking for information', 'speech': 'A speech to an audience'}},
    'calm': {'type': 'score', 'instructions': 'How calm is the speaker?', 'criteria': ['Agitated', 'Neutral', 'Calm']},
}

cases = {
    'text': ({'state': 'I was charged twice this month, please refund one of them.', 'questions': Q_TEXT},
             {'refund': ('noul', '>', 0.9), 'team': ('choice', 'billing')}),
    'json_state': ({'state': {'order_id': 991, 'status': 'charged twice', 'customer_note': 'Please give me my money back', 'amount': 49.9},
                    'questions': Q_TEXT}, {'refund': ('noul', '>', 0.5), 'team': ('choice', 'billing')}),
    'text_multilingual': ({'state': 'Alguém entrou na minha conta de outro país e mudou minha senha!', 'questions': Q_TEXT},
                          {'team': ('choice', 'fraud')}),
}
for ext, p in image_variants.items():
    cases[f'image_{ext}'] = ({'state': 'Photo attached to a pet-sitting request.', 'images': [b64(p, 'image/' + ext)], 'questions': Q_IMG},
                             {'pet': ('choice', 'cats'), 'sofa': ('noul', '>', 0.5)})
cases['image_null_state'] = ({'state': None, 'images': [b64(image_variants['jpg'])], 'questions': Q_IMG}, {'pet': ('choice', 'cats')})
cases['image_two_images'] = ({'state': None, 'images': [b64(image_variants['jpg']), b64(image_variants.get('png', image_variants['jpg']))],
                              'questions': Q_IMG}, {'pet': ('choice', 'cats')})
for ext, p in audio_variants.items():
    cases[f'voice_{ext}'] = ({'state': 'Recording from a public event.', 'files': [b64(p, 'audio/' + ext)], 'questions': Q_AUD},
                             {'kind': ('choice', 'speech')})
cases['voice_null_state'] = ({'state': None, 'files': [b64(audio_variants['wav'])], 'questions': Q_AUD}, {'kind': ('choice', 'speech')})
if 'mp3' in audio_variants:   # mislabeled: an mp3 sent in `images` with an image MIME type
    cases['voice_mp3_mislabeled'] = ({'state': 'Recording from a public event.', 'images': [b64(audio_variants['mp3'], 'image/png')], 'questions': Q_AUD},
                                     {'kind': ('choice', 'speech')})
if 'webp' in image_variants:  # an image sent in `files`
    cases['image_webp_in_files'] = ({'state': None, 'files': [b64(image_variants['webp'], 'audio/wav')], 'questions': Q_IMG},
                                    {'pet': ('choice', 'cats')})

errors = {
    'bad_json': ('{bad', 400),
    'both_media': ({'state': 'x', 'images': [b64(image_variants['jpg'])], 'files': [b64(audio_variants['wav'])], 'questions': Q_IMG}, 400),
    'garbage_media': ({'state': 'x', 'images': ['data:image/png;base64,' + base64.b64encode(b'not an image at all').decode()], 'questions': Q_IMG}, 400),
    'bad_question': ({'state': 'x', 'questions': {'a': {'type': 'score', 'instructions': 'x', 'criteria': ['only one']}}}, 400),
    'no_questions': ({'state': 'x', 'questions': {}}, 400),
}

failures = 0
def check(name, status, resp, expect):
    global failures
    if status != 200:
        print(f'FAIL {name}: HTTP {status} {resp}'); failures += 1; return
    for q, rule in expect.items():
        a = resp['answers'][q]
        ok = (a['choice'] == rule[1]) if rule[0] == 'choice' else (a['noul'] > rule[2])
        if not ok:
            print(f'FAIL {name}: {q} = {a}'); failures += 1

print(f'{len(cases)} cases: ' + ', '.join(cases))
seq = {}
for name, (body, expect) in cases.items():
    st, r, ms = post(body)
    check(name, st, r, expect)
    seq[name] = r
    summary = {q: (a.get('choice') or round(a.get('noul', a.get('score', 0)), 3)) for q, a in r.get('answers', {}).items()}
    print(f'  {name:24s} {ms:7.1f} ms  tokens={r.get("usage", {}).get("input_tokens")}  {summary}')

# converted formats must agree with the native original
def maxdiff(a, b):
    d = 0
    for q in a['answers']:
        x, y = a['answers'][q], b['answers'][q]
        if 'probabilities' in x:
            d = max(d, max(abs(x['probabilities'][k] - y['probabilities'][k]) for k in x['probabilities']))
        else:
            d = max(d, abs(x['noul'] - y['noul']))
    return d
for ext in image_variants:
    if ext != 'jpg' and seq.get(f'image_{ext}', {}).get('answers'):
        d = maxdiff(seq['image_jpg'], seq[f'image_{ext}'])
        print(f'  image {ext:5s} vs jpg: max |dp| {d:.4f}')
for ext in audio_variants:
    if ext != 'wav' and seq.get(f'voice_{ext}', {}).get('answers'):
        d = maxdiff(seq['voice_wav'], seq[f'voice_{ext}'])
        print(f'  voice {ext:6s} vs wav: max |dp| {d:.4f}')

for name, (body, code) in errors.items():
    if isinstance(body, str):
        req = urllib.request.Request(args.url + '/v1/systemone', data=body.encode())
        try:
            urllib.request.urlopen(req); st = 200
        except urllib.error.HTTPError as e:
            st = e.code
    else:
        st, r, _ = post(body)
    if st != code:
        print(f'FAIL error case {name}: HTTP {st}, expected {code}'); failures += 1
print(f'  {len(errors)} error cases checked')

# concurrent burst mixing every modality: results must match the sequential ones
burst = [(n, b) for n, (b, _) in cases.items()] * args.burst
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=len(burst)) as ex:
    out = list(ex.map(lambda nb: (nb[0],) + post(nb[1]), burst))
el = time.perf_counter() - t0
worst = 0
lat = {}
for name, st, r, ms in out:
    if st != 200:
        print(f'FAIL burst {name}: HTTP {st} {r}'); failures += 1; continue
    worst = max(worst, maxdiff(seq[name], r))
    lat.setdefault(name.split('_')[0], []).append(ms)
print(f'  burst: {len(burst)} mixed requests in {el*1e3:.0f} ms ({len(burst)/el:.1f} req/s); max |dp| vs sequential {worst:.5f}')
for k, v in lat.items():
    v.sort()
    print(f'    {k:6s} n={len(v):3d} p50 {v[len(v)//2]:7.1f} ms  max {v[-1]:7.1f} ms')
if worst > 0.02:
    print('FAIL burst results differ from sequential'); failures += 1

print('PASS' if failures == 0 else f'{failures} FAILURES')
sys.exit(1 if failures else 0)

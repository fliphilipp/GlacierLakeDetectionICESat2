"""Create Zenodo drafts, reserve DOIs, upload files, and (only on demand) publish.

The token is read from the ZENODO_TOKEN environment variable and is never printed, logged
or written to disk. Deposition ids and reserved DOIs are kept in a small state file so that
an interrupted upload resumes instead of starting over.

Nothing here publishes unless you pass --yes-publish. A draft is private: you can create it,
reserve its DOI, put the DOI in the manuscript, and upload 11 GB over several days without
anything becoming visible.

    export ZENODO_TOKEN=...                        # deposit:write, deposit:actions
    # or: write it to ~/.config/zenodo/token and chmod 600 that file
    python .../zenodo_push.py create  --record r1 --sandbox
    python .../zenodo_push.py upload  --record r1 --sandbox
    python .../zenodo_push.py status  --record r1 --sandbox
    python .../zenodo_push.py link    --sandbox                 # cross-reference the records
    python .../zenodo_push.py publish --record r1 --sandbox --yes-publish

Always rehearse on --sandbox first. Sandbox DOIs are not real and must not go in the paper.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import requests

PROD = 'https://zenodo.org/api'
SANDBOX = 'https://sandbox.zenodo.org/api'

# The build directory: the one holding metadata/ and the staged record folders. While this
# script lives in zenodo_build/scripts/ the parent of its own directory is correct. Once it is
# archived in the code repo as paper_scripts/ that guess is wrong, so allow an override.
_DEFAULT_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.abspath(os.environ.get('ZENODO_BUILD_DIR') or _DEFAULT_HERE)
META_DIR = os.path.join(HERE, 'metadata')

# record key -> (metadata file, directory of files to upload)
def _check_build_dir():
    if not os.path.isdir(META_DIR):
        sys.exit(f'No metadata directory at {META_DIR}.\n'
                 f'This script resolves the build directory as the parent of its own folder, '
                 f'which is only right while it sits in zenodo_build/scripts/.\n'
                 f'Set ZENODO_BUILD_DIR to the zenodo_build directory and re-run.')


RECORDS = {
    'r1': ('r1_depths.json', os.path.join(HERE, 'greenland')),
    'r2': ('r2_training_models.json', os.path.join(HERE, 'greenland_training')),
}

# Which records point at which, once every DOI is known. Zenodo relation types.
CROSS_LINKS = {
    'r1': [('r2', 'isSupplementedBy')],
    'r2': [('r1', 'isSupplementTo')],
}


def state_path(sandbox):
    return os.path.join(META_DIR, f'.zenodo_state{"_sandbox" if sandbox else ""}.json')


def load_state(sandbox):
    p = state_path(sandbox)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(sandbox, state):
    with open(state_path(sandbox), 'w') as f:
        json.dump(state, f, indent=2)
        f.write('\n')


# Read from the environment first, then from a file outside the repo. The file route exists
# so the token never has to be pasted into a shell command, a transcript or a tracked file.
# A PUT to an existing bucket key replaces the object, so a corrupted transfer can simply
# be re-sent. Observed rate on the 11.6 GB R1 upload: 1 corrupted file in 14.
UPLOAD_ATTEMPTS = 4

TOKEN_FILE = os.path.expanduser('~/.config/zenodo/token')
TOKEN_FILE_SANDBOX = os.path.expanduser('~/.config/zenodo/token_sandbox')

_TOKEN_CACHE = {}


def token(sandbox=False):
    if sandbox in _TOKEN_CACHE:
        return _TOKEN_CACHE[sandbox]
    env_var = 'ZENODO_SANDBOX_TOKEN' if sandbox else 'ZENODO_TOKEN'
    path = TOKEN_FILE_SANDBOX if sandbox else TOKEN_FILE
    t = os.environ.get(env_var)
    if not t and os.path.exists(path):
        mode = os.stat(path).st_mode & 0o777
        if mode & 0o077:
            sys.exit(f'{path} is mode {mode:o}; it holds a credential. chmod 600 it first.')
        t = open(path).read().strip()
    if not t:
        host = 'sandbox.zenodo.org' if sandbox else 'zenodo.org'
        sys.exit(
            f'No token. Create one at https://{host}/account/settings/applications/tokens/new/\n'
            f'with scopes deposit:write and deposit:actions, then either\n'
            f'  export {env_var}=...\n'
            f'or write it to {path} and chmod 600 that file.')
    _TOKEN_CACHE[sandbox] = t
    return t


def api(sandbox):
    return SANDBOX if sandbox else PROD


def request(method, url, sandbox, **kw):
    """One API call with bounded retries on transient failures."""
    kw.setdefault('params', {})['access_token'] = token(sandbox)
    last = None
    for attempt in range(5):
        try:
            r = requests.request(method, url, timeout=kw.pop('timeout', 120), **kw)
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f'HTTP {r.status_code}'
            time.sleep(2 ** attempt * 2)
            continue
        return r
    sys.exit(f'{method} {url.split("?")[0]} failed after retries: {last}')


def die_on_error(r, what):
    if r.status_code >= 400:
        try:
            detail = json.dumps(r.json(), indent=2)[:2000]
        except Exception:
            detail = r.text[:2000]
        sys.exit(f'{what} failed: HTTP {r.status_code}\n{detail}')
    return r.json() if r.text else {}


def md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def files_in(directory):
    """Every file under `directory`, as (relative name, absolute path), sorted, flattened.

    Zenodo presents files as a flat list, so a file in a subdirectory is uploaded under its
    basename. Duplicate basenames across subdirectories would collide, so that is an error.
    """
    out = {}
    for root, _, names in os.walk(directory):
        for n in sorted(names):
            if n.startswith('.'):
                continue
            key = n
            full = os.path.join(root, n)
            if key in out:
                sys.exit(f'name collision when flattening: {full} and {out[key]} '
                         f'both upload as "{key}"')
            out[key] = full
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ commands

def cmd_create(a):
    meta_file, _ = RECORDS[a.record]
    doc = json.load(open(os.path.join(META_DIR, meta_file)))
    meta = dict(doc['metadata'])

    for c in meta.get('creators', []):
        if str(c.get('orcid', '')).startswith('ORCID_'):
            if not a.allow_placeholder_orcid:
                sys.exit(f'creator "{c["name"]}" still has a placeholder ORCID. Fill it in '
                         f'{meta_file}, delete the field, or pass --allow-placeholder-orcid.')
            c.pop('orcid')

    meta['prereserve_doi'] = True
    state = load_state(a.sandbox)
    if a.record in state:
        sys.exit(f'{a.record} already has deposition {state[a.record]["id"]} in '
                 f'{state_path(a.sandbox)}. Delete that entry to start over.')

    r = request('POST', f'{api(a.sandbox)}/deposit/depositions', a.sandbox,
                json={'metadata': meta}, headers={'Content-Type': 'application/json'})
    dep = die_on_error(r, 'create deposition')

    doi = (dep.get('metadata', {}).get('prereserve_doi') or {}).get('doi')
    state[a.record] = {'id': dep['id'], 'bucket': dep['links'].get('bucket'),
                       'doi': doi, 'title': meta['title'], 'sandbox': a.sandbox}
    save_state(a.sandbox, state)

    print(f'created {a.record}: deposition {dep["id"]}')
    print(f'  reserved DOI : {doi}')
    print(f'  edit at      : {dep["links"].get("html")}')
    if a.sandbox:
        print('  NOTE: sandbox DOI. Not real. Do not put it in the manuscript.')


def cmd_upload(a):
    state = load_state(a.sandbox)
    if a.record not in state:
        sys.exit(f'no draft for {a.record}; run `create` first')
    dep = state[a.record]
    _, src_dir = RECORDS[a.record]
    local = files_in(src_dir)

    r = request('GET', f'{api(a.sandbox)}/deposit/depositions/{dep["id"]}/files', a.sandbox)
    remote = {f['filename']: f.get('checksum', '').replace('md5:', '')
              for f in die_on_error(r, 'list files')}

    total = sum(os.path.getsize(p) for p in local.values())
    print(f'{a.record}: {len(local)} files, {total/1e9:.2f} GB · {len(remote)} already uploaded\n')

    done_bytes = 0
    for name, path in local.items():
        size = os.path.getsize(path)
        if name in remote:
            if a.verify:
                local_md5 = md5(path)
                mark = 'ok' if local_md5 == remote[name] else 'CHECKSUM MISMATCH'
                print(f'  [skip] {name:<42} {size/1e6:>9,.1f} MB  {mark}')
                if mark != 'ok':
                    sys.exit(f'    local {local_md5} != remote {remote[name]} - delete the '
                             f'remote file in the Zenodo UI and re-run')
            else:
                print(f'  [skip] {name:<42} {size/1e6:>9,.1f} MB  (already there)')
            done_bytes += size
            continue

        expect = md5(path)
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            t0 = time.time()
            with open(path, 'rb') as fh:
                r = request('PUT', f'{dep["bucket"]}/{name}', a.sandbox, data=fh, timeout=None)
            info = die_on_error(r, f'upload {name}')
            got = info.get('checksum', '').replace('md5:', '')
            dt = time.time() - t0
            if got == expect:
                break
            print(f'  [BAD ] {name:<42} zenodo {got} != local {expect} '
                  f'(attempt {attempt}/{UPLOAD_ATTEMPTS}, re-sending)')
        else:
            sys.exit(f'  upload of {name} still corrupted after {UPLOAD_ATTEMPTS} attempts. '
                     f'The local file is not the problem - verify it with md5 and check the '
                     f'network path before retrying.')
        done_bytes += size
        print(f'  [sent] {name:<42} {size/1e6:>9,.1f} MB  {dt:>6.1f}s '
              f'({size/1e6/max(dt,.01):.1f} MB/s)  checksum ok '
              f'[{done_bytes/total*100:.0f}%]')

    print(f'\n{a.record}: all {len(local)} files present and checksummed.')


def cmd_status(a):
    state = load_state(a.sandbox)
    keys = [a.record] if a.record else list(state)
    for k in keys:
        if k not in state:
            print(f'{k}: no draft'); continue
        dep = state[k]
        r = request('GET', f'{api(a.sandbox)}/deposit/depositions/{dep["id"]}', a.sandbox)
        d = die_on_error(r, 'get deposition')
        files = d.get('files', [])
        nbytes = sum(f.get('filesize', 0) for f in files)
        print(f'{k}: deposition {dep["id"]} · {d.get("state")} · '
              f'submitted={d.get("submitted")} · {len(files)} files · {nbytes/1e9:.2f} GB')
        print(f'   DOI   : {dep.get("doi")}')
        print(f'   title : {d.get("metadata", {}).get("title", "")[:90]}')


def cmd_link(a):
    """Add related_identifiers so each record points at the others. Run after all creates."""
    state = load_state(a.sandbox)
    missing = [k for k in CROSS_LINKS if k not in state]
    if missing:
        sys.exit(f'cannot cross-link: no draft yet for {missing}')

    for rec, links in CROSS_LINKS.items():
        dep = state[rec]
        meta_file, _ = RECORDS[rec]
        meta = dict(json.load(open(os.path.join(META_DIR, meta_file)))['metadata'])
        meta.pop('prereserve_doi', None)
        for c in meta.get('creators', []):
            if str(c.get('orcid', '')).startswith('ORCID_'):
                c.pop('orcid')
        rel = [{'identifier': state[other]['doi'], 'relation': relation,
                'resource_type': 'dataset'}
               for other, relation in links if state.get(other, {}).get('doi')]
        rel += [{'identifier': d, 'relation': r, 'resource_type': t}
                for d, r, t in (a.extra or [])]
        meta['related_identifiers'] = rel
        r = request('PUT', f'{api(a.sandbox)}/deposit/depositions/{dep["id"]}', a.sandbox,
                    json={'metadata': meta}, headers={'Content-Type': 'application/json'})
        die_on_error(r, f'update metadata for {rec}')
        print(f'{rec}: linked to ' + ', '.join(f'{x["identifier"]} ({x["relation"]})'
                                               for x in rel))


def cmd_publish(a):
    state = load_state(a.sandbox)
    if a.record not in state:
        sys.exit(f'no draft for {a.record}')
    dep = state[a.record]
    if not a.yes_publish:
        print(f'WOULD publish {a.record}: deposition {dep["id"]}, DOI {dep["doi"]}')
        print(f'  "{dep["title"]}"')
        print('\nPublishing is permanent: the files and metadata become public and the files '
              'of a published record cannot be changed (a new version can be added).')
        print('Re-run with --yes-publish to go ahead.')
        return
    r = request('POST', f'{api(a.sandbox)}/deposit/depositions/{dep["id"]}/actions/publish',
                a.sandbox)
    d = die_on_error(r, 'publish')
    print(f'published {a.record}: {d.get("doi_url") or d.get("doi")}')


def main():
    # --sandbox lives on a parent parser so it is accepted both before and after the
    # subcommand. Getting it silently wrong would mean minting a real DOI while rehearsing.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--sandbox', action='store_true',
                        help='target sandbox.zenodo.org instead of the real one')

    p = argparse.ArgumentParser(description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('create', parents=[common], help='create a draft and reserve its DOI')
    c.add_argument('--record', required=True, choices=RECORDS)
    c.add_argument('--allow-placeholder-orcid', action='store_true')
    c.set_defaults(func=cmd_create)

    u = sub.add_parser('upload', parents=[common], help='upload files to a draft, resuming and checksumming')
    u.add_argument('--record', required=True, choices=RECORDS)
    u.add_argument('--verify', action='store_true',
                   help='also re-checksum files already uploaded (slow, reads every byte)')
    u.set_defaults(func=cmd_upload)

    s = sub.add_parser('status', parents=[common], help='show draft state')
    s.add_argument('--record', choices=RECORDS)
    s.set_defaults(func=cmd_status)

    l = sub.add_parser('link', parents=[common], help='cross-reference the records by DOI')
    l.add_argument('--extra', nargs=3, action='append', metavar=('DOI', 'RELATION', 'TYPE'),
                   help='additional related identifier, repeatable')
    l.set_defaults(func=cmd_link)

    b = sub.add_parser('publish', parents=[common], help='publish a draft (permanent)')
    b.add_argument('--record', required=True, choices=RECORDS)
    b.add_argument('--yes-publish', action='store_true')
    b.set_defaults(func=cmd_publish)

    a = p.parse_args()
    _check_build_dir()
    a.func(a)


if __name__ == '__main__':
    main()

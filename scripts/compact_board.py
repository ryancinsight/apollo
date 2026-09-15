"""Compact the apollo board to its budget (APOLLO-BOARD-COMPACTION).

Every `## ` heading is an item. Each keeps (or gains) a stable anchor,
a status from the closed set (todo, in-progress, blocked, review, done),
and a record within the budget: an open item at most fifteen lines, a done
item one line (anchor, identity, outcome) in the closing section.
Narrative beyond that is dropped — git is the archive, and the commit that
lands this compaction names the pre-compaction revision. A stale
in-progress claim (last update before the given day) is released to todo.
Sprint and session sections (the report-file genre inside the board) are
dropped whole.

Usage: python compact_board.py <in> <out> <today> [<release-before>]
Prints the before/after line counts and the anchor sets' difference.
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

src, dst, today = sys.argv[1], sys.argv[2], sys.argv[3]
release_before = sys.argv[4] if len(sys.argv) > 4 else today
CLOSED = ('todo', 'in-progress', 'blocked', 'review', 'done')
STATUS_MAP = {'downgraded': 'done', 'measured': 'done', 'superseded': 'done', 'closed': 'done',
              'landed': 'done', 'merged': 'done', 'in_progress': 'in-progress', 'wip': 'in-progress'}
SECTION = re.compile(r'^## (Closed in this sprint|Open in this sprint|Planned next|Delivered|Sprint|Session|Wave|Tier|Watchpoints|Backlog|Open items|Done items|Notes)', re.I)
GLYPHS = re.compile('[←-⇿⌀-⏿─-➿⬀-⯿\U0001f000-\U0001faff]')
ANCHOR = re.compile(r'^<a id="([^"]+)"></a>\s*$')
NL = chr(10)

text = open(src, encoding='utf-8').read()
lines = text.split(NL)

# ---- the head, then the items: a heading with the anchor line before it.
head = []
i = 0
while i < len(lines) and not lines[i].startswith('## '):
    head.append(lines[i])
    i += 1
pending_anchor = None
while head and (not head[-1].strip() or ANCHOR.match(head[-1])):
    last = head.pop()
    m = ANCHOR.match(last)
    if m:
        pending_anchor = m.group(1)
items = []
current = None
for line in lines[i:]:
    if line.startswith('## '):
        if current:
            items.append(current)
        current = {'anchor': pending_anchor, 'heading': line, 'body': []}
        pending_anchor = None
        continue
    m = ANCHOR.match(line)
    if m:
        pending_anchor = m.group(1)
        continue
    if current is not None:
        current['body'].append(line)
if current:
    items.append(current)

# ---- anchors: existing ones stay; a missing one is the ID lowercased
# without its trailing date, kept unique.
used = {it['anchor'] for it in items if it['anchor']}


def derive(heading):
    m = re.match(r'^## ([A-Z0-9][A-Z0-9-]*)', heading)
    base = (m.group(1) if m else heading[3:20]).lower()
    base = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', base)
    cand, n = base, 2
    while cand in used:
        cand, n = f'{base}-{n}', n + 1
    used.add(cand)
    return cand


for it in items:
    if not it['anchor']:
        it['anchor'] = derive(it['heading'])


def status_of(heading):
    m = re.search(r' — ([A-Za-z_-]+)(\s+\d{4}-\d{2}(-\d{2})?)?\s*$', heading)
    return (m.group(1).lower(), m.group(2)) if m else (None, None)


def records_of(body):
    """The top-level record lines: bullets with continuation lines folded
    in; blank lines dropped; a sub-heading ends the record."""
    out = []
    for line in body:
        if not line.strip():
            continue
        if re.match(r'^#{3,} ', line):
            break
        if line.startswith('- ') or line.startswith('* '):
            out.append(line)
        elif out and (line.startswith('  ') or line.startswith('\t')):
            out[-1] = out[-1] + ' ' + line.strip()
        else:
            out.append(line)
    return out


released, normalized, dropped, done_lines = [], [], [], []
open_lines = []
for it in items:
    heading = GLYPHS.sub('', it['heading']).replace('  ', ' ').rstrip()
    status, date = status_of(heading)
    body_text = ' '.join(line for line in it['body'] if line.strip())
    checklist_only = bool(body_text.strip()) and all(
        re.match(r'^\s*(- \[[x ]\]|\* \[[x ]\])', line) for line in it['body'] if line.strip())
    if status is None and (SECTION.match(heading) or (checklist_only and not re.match(r'^## [A-Z0-9][A-Z0-9-]{6,} ', heading))):
        dropped.append(heading[3:80])
        continue
    if status is None:
        landed = re.search(r'PR #\d+|pull/\d+|\b[0-9a-f]{8,40}\b|[Ll]anded|[Mm]erged|\[x\]|Outcome:\*\* built', body_text[:600])
        status = 'done' if landed else 'todo'
        heading = heading + ' — ' + status
        normalized.append((it['anchor'], f'none -> {status}'))
    elif status not in CLOSED:
        new = STATUS_MAP.get(status, 'todo')
        heading = re.sub(r' — ' + re.escape(status) + r'(\s+\d{4}-\d{2}(-\d{2})?)?\s*$', ' — ' + new, heading)
        normalized.append((it['anchor'], f'{status} -> {new}'))
        status = new
    elif date:
        heading = re.sub(r'(\s+\d{4}-\d{2}(-\d{2})?)\s*$', '', heading)
    records = records_of(it['body'])
    if status == 'in-progress':
        m = re.search(r'last-update:\*\*\s*(\d{4}-\d{2}-\d{2})', ' '.join(records))
        last = m.group(1) if m else None
        if last is not None and last < release_before:
            heading = heading.replace(' — in-progress', ' — todo')
            records = [r for r in records if 'Integrator:' not in r]
            records.append(f'- **Claim released:** {today}, the last update {last} older than the stale-claim window; reclaim by re-syncing the board.')
            released.append((it['anchor'], last))
            status = 'todo'
    if status == 'done':
        m = re.match(r'^## (\S+) — (.*) — done$', heading)
        ident, title = (m.group(1), m.group(2)) if m else (heading[3:], '')
        record = re.sub(r'^[-*] ', '', records[0]).strip() if records else ''
        done_lines.append(f'<a id="{it["anchor"]}"></a>- **{ident}** — {title}. {record}')
        continue
    open_lines.append(f'<a id="{it["anchor"]}"></a>')
    open_lines.append(heading)
    open_lines.extend(records[:14])
    open_lines.append('')

result = head + [''] + open_lines + ['# Done', '', 'One line an item: the anchor, the identity, the outcome with its commit or PR; the narrative lives in git.', ''] + done_lines
out = NL.join(result).rstrip(NL) + NL
open(dst, 'w', encoding='utf-8', newline=NL).write(out)
before_anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
after_anchors = set(re.findall(r'<a id="([^"]+)"></a>', out))
print(f'lines {len(lines)} -> {out.count(NL)}; items {len(items)}; open {len(open_lines)} lines; done {len(done_lines)}; anchors {len(before_anchors)} -> {len(after_anchors)}; lost {sorted(before_anchors - after_anchors)}')
print('released claims:', released)
print('normalized statuses:', len(normalized), normalized[:6])
print('dropped sections:', len(dropped), dropped[:8])

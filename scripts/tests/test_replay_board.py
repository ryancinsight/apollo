"""Board replay: a commit's item changes re-applied onto an upstream board that
moved underneath it, checked against the board both commits produce together,
and end to end through a real rebase conflict."""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay_board = load("replay_board")
compaction = load("compact_board")
TODAY = "2026-09-17"


def item(ident: str, status: str = "todo", record: str | None = None) -> str:
    anchor = ident.lower()
    return f'<a id="{anchor}"></a>\n## {ident} — Item {anchor} [patch] — {status}\n- {record or f"**Scope:** {anchor}."}\n\n'


def board(*items: str) -> str:
    return compaction.compact("# Backlog — sample\n\n" + "".join(items), TODAY, TODAY)[0]


LANDED = [item(f"LANDED-{n}", "done", f"[#{n}](https://example.invalid/pull/{n}): landed.") for n in range(6)]
BASE = board(*LANDED, item("FIRST"), item("MIDDLE"), item("LAST"))
# The commit being replayed: records FIRST as done and claims LAST.
PARENT = BASE
CHILD = board(
    *LANDED,
    item("FIRST", "done", "[#10](https://example.invalid/pull/10): first landed."),
    item("MIDDLE"),
    item("LAST", "in-progress", "**Integrator:** me; **last-update:** 2026-09-17."),
)
# Upstream moved first: records MIDDLE and files a new item after it.
UPSTREAM = board(
    *LANDED,
    item("FIRST"),
    item("MIDDLE", "done", "[#11](https://example.invalid/pull/11): middle landed."),
    item("FILED"),
    item("LAST"),
)
BOTH = board(
    *LANDED,
    item("FIRST", "done", "[#10](https://example.invalid/pull/10): first landed."),
    item("MIDDLE", "done", "[#11](https://example.invalid/pull/11): middle landed."),
    item("FILED"),
    item("LAST", "in-progress", "**Integrator:** me; **last-update:** 2026-09-17."),
)


def replayed(working: str, parent: str, child: str) -> tuple[str, int]:
    target = replay_board.Board(working)
    changed = replay_board.replay(target, replay_board.Board(parent), replay_board.Board(child))
    return compaction.compact(target.render(), TODAY, TODAY)[0], changed


class Replay(unittest.TestCase):
    def test_reproduces_the_board_both_commits_make(self):
        out, changed = replayed(UPSTREAM, PARENT, CHILD)
        self.assertEqual(changed, 3)  # FIRST's block and entry, LAST's block
        self.assertEqual(out, BOTH)

    def test_replaying_onto_its_own_parent_reproduces_the_commit(self):
        out, _ = replayed(PARENT, PARENT, CHILD)
        self.assertEqual(out, CHILD)

    def test_an_unchanged_commit_changes_nothing(self):
        out, changed = replayed(UPSTREAM, PARENT, PARENT)
        self.assertEqual(changed, 0)
        self.assertEqual(out, UPSTREAM)

    def test_a_new_block_follows_its_predecessor(self):
        child = board(*LANDED, item("FIRST"), item("ADDED"), item("MIDDLE"), item("LAST"))
        out, _ = replayed(UPSTREAM, PARENT, child)
        opened = out.split("# Done")[0]
        self.assertLess(opened.index("## FIRST "), opened.index("## ADDED "))
        self.assertLess(opened.index("## ADDED "), opened.index("## FILED "))

    def test_a_block_the_commit_removed_is_removed(self):
        self.assertIn("## LAST ", UPSTREAM)  # still open upstream, so removal is the replay's doing
        child = board(*LANDED, item("FIRST"), item("MIDDLE"))
        out, _ = replayed(UPSTREAM, PARENT, child)
        self.assertNotIn("## LAST ", out)
        self.assertNotIn('id="last"', out)
        self.assertIn("## FILED ", out)


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class RebaseConflict(unittest.TestCase):
    def git(self, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, check=False)
        return result.stdout.decode("utf-8").strip()

    def commit(self, text: str, message: str) -> None:
        (self.repo / "backlog.md").write_text(text, encoding="utf-8", newline="\n")
        self.git("add", "backlog.md")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "core.autocrlf", "false")
        self.commit(BASE, "base")
        self.git("switch", "-q", "-c", "topic")
        self.commit(CHILD, "record first")
        self.git("switch", "-q", "main")
        self.commit(UPSTREAM, "record middle")

    def tearDown(self):
        self.directory.cleanup()

    def test_resolves_the_conflict_a_rebase_stops_on(self):
        self.git("switch", "-q", "topic")
        rebase = subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "rebase", "main"],
            cwd=self.repo, capture_output=True, check=False,
        )
        self.assertNotEqual(rebase.returncode, 0, "the fixture must conflict for the test to mean anything")
        self.git("checkout", "--ours", "--", "backlog.md")
        head = self.git("rev-parse", "REBASE_HEAD")
        script = Path(__file__).parents[1] / "replay_board.py"
        run = subprocess.run(
            [sys.executable, str(script), str(self.repo / "backlog.md"), head],
            capture_output=True, check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8"))
        text = (self.repo / "backlog.md").read_text(encoding="utf-8")
        self.assertEqual(compaction.compact(text, TODAY, TODAY)[0], BOTH)


if __name__ == "__main__":
    unittest.main()

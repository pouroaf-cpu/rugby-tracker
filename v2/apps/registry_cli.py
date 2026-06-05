"""registry_cli.py - inspect / manage the persistent player registry.

Small argparse CLI over rt2.registry.PlayerRegistry. Examples:
    python v2/apps/registry_cli.py --list
    python v2/apps/registry_cli.py --set-me Pou --number 7
    python v2/apps/registry_cli.py --show <uuid>
    python v2/apps/registry_cli.py --rename <uuid> "New Name"
    python v2/apps/registry_cli.py --ruleout 7
    python v2/apps/registry_cli.py --ingest output/match.game.profiles.json --match-id match_x
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # adds v2/
from rt2.paths import ProjectPaths
from rt2.profiles import GameProfiles
from rt2.registry import PlayerRegistry, registry_db


def _print_players(reg):
    players = reg.list_players()
    if not players:
        print("(no players in registry)")
        return
    print(f"{'uuid':<34} {'name':<16} {'#':>3} {'role':<20} "
          f"{'me':>2} {'frm':>6} {'mtch':>4}")
    print("-" * 92)
    for p in players:
        num = p["current_number"]
        print(f"{p['uuid']:<34} {(p['display_name'] or ''):<16} "
              f"{(str(num) if num is not None else '-'):>3} "
              f"{(p['role'] or ''):<20} {('Y' if p['is_me'] else ''):>2} "
              f"{int(p['n_frames'] or 0):>6} {int(p['n_matches'] or 0):>4}")


def _print_show(reg, uuid):
    p = reg.get_player(uuid)
    if p is None:
        print(f"no such player: {uuid}")
        return
    sig = reg.get_signature(uuid) or {}
    print(f"uuid          : {p['uuid']}")
    print(f"display_name  : {p['display_name']}")
    print(f"is_me         : {p['is_me']}   is_teammate: {p['is_teammate']}")
    print(f"current_number: {p['current_number']}   role: {p['role']}")
    print(f"jersey_history: {p['jersey_history']}")
    print(f"n_frames={p['n_frames']}  n_clips={p['n_clips']}  n_matches={p['n_matches']}")
    print(f"height_resid  : mean={sig.get('height_resid_mean')} "
          f"std={sig.get('height_resid_std')} n={sig.get('height_n')}")
    print(f"zone          : ({sig.get('zone_x')}, {sig.get('zone_y')})  "
          f"speed={sig.get('speed_mean')}")
    print(f"fingerprint_n : {sig.get('fingerprint_n')}")
    print(f"number_votes  : {sig.get('number_votes')}")


def _print_ruleout(reg, key):
    rows = reg.rule_out_for(key)
    if not rows:
        print("(no other teammates to rule out)")
        return
    print(f"rule-out seed for {key!r}: {len(rows)} other teammate(s)")
    for r in rows:
        h = r["height_resid_mean"]
        h_s = f"{h:+.1f}px" if h is not None else "  ?  "
        print(f"  #{(str(r['number']) if r['number'] is not None else '-'):>3} "
              f"{(r['display_name'] or ''):<16} {(r['role'] or ''):<20} "
              f"h{h_s}  ev={r['evidence']}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Inspect/manage the player registry.")
    ap.add_argument("--db", default=None, help="path to registry.sqlite (default: output/)")
    ap.add_argument("--list", action="store_true", help="list all players")
    ap.add_argument("--show", metavar="UUID", help="show one player's full record")
    ap.add_argument("--set-me", metavar="NAME", help="create/mark the 'me' player")
    ap.add_argument("--number", type=int, default=None, help="jersey number (with --set-me)")
    ap.add_argument("--rename", nargs=2, metavar=("UUID", "NAME"), help="rename a player")
    ap.add_argument("--ruleout", metavar="NUM_OR_UUID", help="print rule-out list for a player")
    ap.add_argument("--ingest", metavar="PROFILES_JSON",
                    help="ingest a GameProfiles JSON file")
    ap.add_argument("--match-id", default=None, help="match id for --ingest")
    args = ap.parse_args(argv)

    db = args.db or registry_db(ProjectPaths())
    with PlayerRegistry(db) as reg:
        did = False
        if args.set_me:
            uuid = reg.set_me(args.set_me, number=args.number)
            print(f"me = {uuid} ({args.set_me}"
                  f"{', #' + str(args.number) if args.number is not None else ''})")
            did = True
        if args.rename:
            reg.rename(args.rename[0], args.rename[1])
            print(f"renamed {args.rename[0]} -> {args.rename[1]}")
            did = True
        if args.ingest:
            gp = GameProfiles.load(args.ingest)
            summary = reg.ingest_game_profiles(
                gp, match_id=args.match_id, video=getattr(gp, "game", ""))
            print(f"ingested: {summary}")
            did = True
        if args.show:
            _print_show(reg, args.show)
            did = True
        if args.ruleout:
            _print_ruleout(reg, args.ruleout)
            did = True
        if args.list or not did:
            _print_players(reg)


if __name__ == "__main__":
    main()

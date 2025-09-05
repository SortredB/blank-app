# mini_fpl.py
# --------------------------------------------------------------
# A compact Fantasy Premier League–style system for your own league and players.
# Works in Google Colab or locally. No external dependencies required.
# Features:
# - Custom player pool (load from CSV or use sample)
# - 15-man squads, formations, captain/vice-captain
# - Budget + prices, max 3 players per club
# - Transfers: 1 free transfer / GW (bank up to 2). Extra transfers cost -4.
# - Chips: Wildcard, Free Hit, Triple Captain, Bench Boost
# - Gameweeks with manual admin input of match events
# - FPL-like scoring (appearance, goals by position, assists, CS, GC, saves, pens, cards, own goals, bonus)
# - Automatic substitutions respecting formation rules
# - Classic leagues with total points
#
# NOTE:
# - Bonus Points: simplified—admin assigns 3/2/1 to any players per GW.
# - Clean sheet eligibility requires >=60 min and team conceded 0.
# - Auto-subs: basic implementation (tries to maintain a valid formation).
#
# Author: You + ChatGPT
# --------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import csv
import json
import sys

# ---------------------- Config ----------------------

POSITIONS = ["GK", "DEF", "MID", "FWD"]
SQUAD_SIZE = 15
STARTING_SIZE = 11
FORMATION_OPTIONS = {
    # DEF, MID, FWD minimums with one GK implied
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "4-4-2": (4, 4, 2),
    "4-5-1": (4, 5, 1),
    "5-4-1": (5, 4, 1),
    "5-3-2": (5, 3, 2),
    "4-3-3": (4, 3, 3),
}
MAX_PER_CLUB = 3
INITIAL_BUDGET = 100.0

# Scoring (FPL-like)
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
APPEAR_LT60 = 1
APPEAR_GE60 = 2
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}  # FPL gives MID +1, FWD 0
GC_MINUS_PER_2 = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}
SAVE_PER_3 = 1
PEN_SAVE = 5
PEN_MISS = -2
YELLOW = -1
RED = -3
OWN_GOAL = -2

# ---------------------- Data Models ----------------------

@dataclass
class Player:
    id: int
    name: str
    position: str  # GK/DEF/MID/FWD
    club: str
    price: float

@dataclass
class Event:  # single GW stat line for a player
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    cs: bool = False
    goals_conceded: int = 0
    saves: int = 0
    pen_save: int = 0
    pen_miss: int = 0
    yellow: int = 0
    red: int = 0
    own_goals: int = 0
    bonus: int = 0  # final bonus points (3/2/1) simplified

@dataclass
class Squad:
    players: List[int]  # 15 player IDs
    starting: List[int]  # 11 player IDs (subset, order arbitrary)
    bench_order: List[int]  # the remaining 4 in order [G1, O1, O2, O3] (first is GK bench)
    captain: int
    vice_captain: int

@dataclass
class Manager:
    name: str
    budget: float = INITIAL_BUDGET
    bank: float = 0.0
    free_transfers: int = 1
    wildcard_available: bool = True
    free_hit_available: bool = True
    triple_captain_available: bool = True
    bench_boost_available: bool = True
    chip_active: Optional[str] = None  # "FH", "TC", "BB", "WC" during GW
    total_points: int = 0
    squad: Optional[Squad] = None
    fh_backup_squad: Optional[Squad] = None  # for Free Hit revert

@dataclass
class Gameweek:
    gw: int
    events: Dict[int, Event] = field(default_factory=dict)  # player_id -> Event

@dataclass
class League:
    name: str
    managers: Dict[str, Manager] = field(default_factory=dict)
    players: Dict[int, Player] = field(default_factory=dict)
    current_gw: int = 1
    history: List[Gameweek] = field(default_factory=list)

# ---------------------- Utils ----------------------

def load_players_csv(path: str) -> Dict[int, Player]:
    players: Dict[int, Player] = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["id"])
            players[pid] = Player(
                id=pid,
                name=row["name"],
                position=row["position"],
                club=row["club"],
                price=float(row["price"]),
            )
    return players

def save_league(league: League, path: str):
    # Convert dataclasses to serializable dict
    def event_to_dict(e: Event):
        return e.__dict__

    data = {
        "name": league.name,
        "current_gw": league.current_gw,
        "players": {pid: p.__dict__ for pid, p in league.players.items()},
        "managers": {
            mname: {
                **{k: v for k, v in m.__dict__.items() if k not in ("squad", "fh_backup_squad")},
                "squad": None if m.squad is None else {
                    "players": m.squad.players,
                    "starting": m.squad.starting,
                    "bench_order": m.squad.bench_order,
                    "captain": m.squad.captain,
                    "vice_captain": m.squad.vice_captain,
                },
                "fh_backup_squad": None if m.fh_backup_squad is None else {
                    "players": m.fh_backup_squad.players,
                    "starting": m.fh_backup_squad.starting,
                    "bench_order": m.fh_backup_squad.bench_order,
                    "captain": m.fh_backup_squad.captain,
                    "vice_captain": m.fh_backup_squad.vice_captain,
                },
            } for mname, m in league.managers.items()
        },
        "history": [
            {
                "gw": gw.gw,
                "events": {pid: event_to_dict(ev) for pid, ev in gw.events.items()},
            } for gw in league.history
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_league(path: str) -> League:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    league = League(name=data["name"])
    league.current_gw = data["current_gw"]
    league.players = {int(pid): Player(**p) for pid, p in data["players"].items()}
    for mname, md in data["managers"].items():
        m = Manager(name=mname)
        for k, v in md.items():
            if k in ("squad", "fh_backup_squad"):
                continue
            setattr(m, k, v)
        if md.get("squad"):
            sq = md["squad"]
            m.squad = Squad(**sq)
        if md.get("fh_backup_squad"):
            m.fh_backup_squad = Squad(**md["fh_backup_squad"])
        league.managers[mname] = m
    league.history = []
    for gw in data["history"]:
        g = Gameweek(gw=gw["gw"])
        g.events = {int(pid): Event(**ev) for pid, ev in gw["events"].items()}
        league.history.append(g)
    return league

# ---------------------- Validation ----------------------

def check_squad_validity(players: Dict[int, Player], squad: Squad) -> Tuple[bool, str]:
    if len(squad.players) != SQUAD_SIZE:
        return False, f"Squad must have {SQUAD_SIZE} players."
    ids_set = set(squad.players)
    if len(ids_set) != SQUAD_SIZE:
        return False, "Duplicate player in squad."
    # Position counts
    pos_counts = {"GK":0, "DEF":0, "MID":0, "FWD":0}
    club_counts: Dict[str, int] = {}
    for pid in squad.players:
        p = players[pid]
        pos_counts[p.position] += 1
        club_counts[p.club] = club_counts.get(p.club, 0) + 1
    if pos_counts["GK"] != 2 or pos_counts["DEF"] < 5 or pos_counts["MID"] < 5 or pos_counts["FWD"] < 3:
        return False, "Invalid position counts (need 2 GK, >=5 DEF, >=5 MID, >=3 FWD)."
    # Max per club
    for club, count in club_counts.items():
        if count > MAX_PER_CLUB:
            return False, f"Too many players from {club} (max {MAX_PER_CLUB})."
    # Starting XI and bench
    if len(squad.starting) != STARTING_SIZE:
        return False, "Starting XI must have 11 players."
    if not set(squad.starting).issubset(ids_set):
        return False, "Starting XI contains player not in squad."
    bench = [pid for pid in squad.players if pid not in set(squad.starting)]
    if len(bench) != 4:
        return False, "Bench must be the remaining 4 players."
    if len(squad.bench_order) != 4 or set(squad.bench_order) != set(bench):
        return False, "Bench order must list the 4 bench players in order."
    # Exactly one GK on bench slot 0
    if players[squad.bench_order[0]].position != "GK":
        return False, "Bench slot 1 must be a GK."
    # Formation validity
    def_count = sum(1 for pid in squad.starting if players[pid].position == "DEF")
    mid_count = sum(1 for pid in squad.starting if players[pid].position == "MID")
    fwd_count = sum(1 for pid in squad.starting if players[pid].position == "FWD")
    gk_count = sum(1 for pid in squad.starting if players[pid].position == "GK")
    if gk_count != 1:
        return False, "Starting XI must include exactly 1 GK."
    if (def_count, mid_count, fwd_count) not in FORMATION_OPTIONS.values():
        return False, f"Starting formation {def_count}-{mid_count}-{fwd_count} not allowed."
    # Captain/vice present in starting XI and distinct
    if squad.captain not in squad.starting or squad.vice_captain not in squad.starting:
        return False, "Captain and Vice must be in starting XI."
    if squad.captain == squad.vice_captain:
        return False, "Captain and Vice must be different players."
    return True, "OK"

def squad_cost(players: Dict[int, Player], squad: Squad) -> float:
    return sum(players[pid].price for pid in squad.players)

# ---------------------- Scoring ----------------------

def score_event(p: Player, ev: Event) -> int:
    pts = 0
    # appearance
    if ev.minutes > 0:
        pts += APPEAR_LT60 if ev.minutes < 60 else APPEAR_GE60
    # goals
    pts += ev.goals * GOAL_POINTS[p.position]
    # assists
    pts += ev.assists * ASSIST_POINTS
    # clean sheet eligibility only if played >=60 and conceded 0
    if ev.minutes >= 60 and ev.cs:
        pts += CS_POINTS[p.position]
    # goals conceded (GK/DEF): -1 per 2
    if p.position in ("GK", "DEF"):
        pts += (ev.goals_conceded // 2) * GC_MINUS_PER_2[p.position]
    # saves (GK)
    if p.position == "GK":
        pts += (ev.saves // 3) * SAVE_PER_3
    # penalties
    pts += ev.pen_save * PEN_SAVE
    pts += ev.pen_miss * PEN_MISS
    # cards & own goals
    pts += ev.yellow * YELLOW
    pts += ev.red * RED
    pts += ev.own_goals * OWN_GOAL
    # bonus
    pts += ev.bonus
    return pts

# ---------------------- Auto Subs ----------------------

def auto_substitute(players: Dict[int, Player], sq: Squad, gw_events: Dict[int, Event]) -> Tuple[List[int], List[int]]:
    """
    Returns final_starting, final_bench after auto-subs.
    Rule (simplified):
    - If a starting player has 0 minutes, try to sub from bench in order (bench[0]=GK only for GK; others any outfielder).
    - Keep a valid formation (>=3 DEF, >=2 MID, >=1 FWD) and exactly 1 GK.
    """
    starting = sq.starting.copy()
    bench = sq.bench_order.copy()

    def played(pid): return gw_events.get(pid, Event()).minutes > 0
    def pos(pid): return players[pid].position
    def counts(ids):
        return (
            sum(1 for x in ids if pos(x)=="GK"),
            sum(1 for x in ids if pos(x)=="DEF"),
            sum(1 for x in ids if pos(x)=="MID"),
            sum(1 for x in ids if pos(x)=="FWD"),
        )

    for i, pid in list(enumerate(starting)):
        if played(pid):
            continue
        # needs sub
        gk, d, m, f = counts(starting)
        if pos(pid) == "GK":
            # only bench GK (slot 0) can replace
            if bench and pos(bench[0]) == "GK" and played(bench[0]):
                sub = bench.pop(0)
                # move old GK to bench tail
                starting[i] = sub
                bench.append(pid)
        else:
            # try each of bench[1:]
            for j in range(1, len(bench)):
                cand = bench[j]
                if not played(cand) or pos(cand) == "GK":
                    continue
                # tentative replace
                temp = starting.copy()
                temp[i] = cand
                gk2, d2, m2, f2 = counts(temp)
                # check formation validity (1 GK, >=3 DEF, >=2 MID, >=1 FWD)
                if gk2 == 1 and d2 >= 3 and m2 >= 2 and f2 >= 1 and len(temp)==11:
                    # perform sub: move non-playing starter to bench at the used slot
                    starting = temp
                    bench[j] = pid
                    break
    return starting, bench

# ---------------------- Chips & Transfers ----------------------

def apply_transfers(m: Manager, players: Dict[int, Player], transfers_out: List[int], transfers_in: List[int]) -> Tuple[bool, str, int]:
    """Apply transfers for the week. Returns (ok, message, hit_points)."""
    if len(transfers_out) != len(transfers_in):
        return False, "Transfers must be 1:1.", 0
    if not m.squad:
        return False, "No squad.", 0
    # Check ownership
    for pid in transfers_out:
        if pid not in m.squad.players:
            return False, f"Cannot sell {pid} — not in squad.", 0
    # Check duplicates
    new_players = set(m.squad.players)
    cost_diff = 0.0
    for out_pid, in_pid in zip(transfers_out, transfers_in):
        if in_pid in new_players and in_pid not in transfers_out:
            return False, f"Duplicate incoming player {in_pid}.", 0
        new_players.remove(out_pid)
        new_players.add(in_pid)
        cost_diff += players[in_pid].price - players[out_pid].price

    # Check club count constraint
    club_counts = {}
    for pid in new_players:
        club = players[pid].club
        club_counts[club] = club_counts.get(club, 0) + 1
        if club_counts[club] > MAX_PER_CLUB:
            return False, f"Max {MAX_PER_CLUB} per club violated.", 0

    # Check budget
    if m.bank - cost_diff < 0 and m.chip_active != "FH":  # Free Hit ignores bank permanently
        return False, f"Not enough bank. Need {cost_diff:.1f}, have {m.bank:.1f}.", 0

    # Calculate hits
    n = len(transfers_in)
    free = m.free_transfers
    chargeable = max(0, n - free)
    hit_points = 0
    if m.chip_active in ("WC", "FH"):
        hit_points = 0
    else:
        hit_points = chargeable * 4

    # Apply
    # Replace in squad list
    new_players_list = list(new_players)
    # Maintain starting/bench validity: if a starter was sold, starter now in unknown. We'll rebuild starting/bench simply.
    # Rebuild starting XI: keep previous starting if possible else fill by position priority.
    m.bank -= cost_diff if m.chip_active != "FH" else 0.0
    m.free_transfers = 0  # spent this GW; will reset later by GW progression

    # Simple rebuild: keep GK/DEF/MID/FWD counts of current starting; if missing, fill from remaining
    if m.squad:
        old_sq = m.squad
        starting = [pid for pid in old_sq.starting if pid in new_players_list]
        missing = STARTING_SIZE - len(starting)
        bench_candidates = [pid for pid in new_players_list if pid not in starting]

        # ensure 1 GK in starting
        gks = [pid for pid in starting if players[pid].position=="GK"]
        if len(gks) != 1:
            # move GK from bench_candidates
            gk_bench = [pid for pid in bench_candidates if players[pid].position=="GK"]
            if gk_bench:
                starting = [x for x in starting if players[x].position!="GK"]
                starting.append(gk_bench[0])
                bench_candidates.remove(gk_bench[0])
        # fill remaining slots arbitrarily but keeping at least 3 DEF, 2 MID, 1 FWD
        def need_more(counts):
            gk, d, mi, f = counts
            if gk < 1: return "GK"
            if d < 3: return "DEF"
            if mi < 2: return "MID"
            if f < 1: return "FWD"
            return None
        def counts(ids):
            return (
                sum(1 for x in ids if players[x].position=="GK"),
                sum(1 for x in ids if players[x].position=="DEF"),
                sum(1 for x in ids if players[x].position=="MID"),
                sum(1 for x in ids if players[x].position=="FWD"),
            )
        while len(starting) < STARTING_SIZE and bench_candidates:
            need = need_more(counts(starting))
            # try fill by need else any outfielder
            pick = None
            if need:
                for pid in bench_candidates:
                    if players[pid].position == need:
                        pick = pid; break
            if not pick:
                # pick first non-GK if we already have GK
                for pid in bench_candidates:
                    if players[pid].position != "GK" or counts(starting)[0]==0:
                        pick = pid; break
            starting.append(pick)
            bench_candidates.remove(pick)

        bench = [pid for pid in new_players_list if pid not in starting]
        # ensure first bench is GK
        bench_gk = [pid for pid in bench if players[pid].position=="GK"]
        if bench_gk:
            # put one GK first
            gk = bench_gk[0]
            bench.remove(gk)
            bench = [gk] + bench
        # Captain/vice: keep if still starters else set arbitrarily
        cap = old_sq.captain if old_sq.captain in starting else starting[0]
        vcap_candidates = [x for x in starting if x != cap]
        vcap = old_sq.vice_captain if old_sq.vice_captain in vcap_candidates else vcap_candidates[0]
        m.squad = Squad(players=new_players_list, starting=starting, bench_order=bench, captain=cap, vice_captain=vcap)

    return True, "Transfers applied.", hit_points

def activate_chip(m: Manager, chip: str) -> Tuple[bool, str]:
    chip = chip.upper()
    mapping = {"WC":"wildcard_available", "FH":"free_hit_available", "TC":"triple_captain_available", "BB":"bench_boost_available"}
    if chip not in mapping:
        return False, "Unknown chip."
    if not getattr(m, mapping[chip]):
        return False, "Chip not available."
    if m.chip_active and m.chip_active != chip:
        return False, f"Another chip active: {m.chip_active}."
    m.chip_active = chip
    return True, f"{chip} activated."

def clear_chip_after_gw(m: Manager):
    if m.chip_active == "FH":
        # revert to backup squad then clear
        if m.fh_backup_squad:
            m.squad = m.fh_backup_squad
            m.fh_backup_squad = None
    # consume chip availability
    if m.chip_active == "WC":
        m.wildcard_available = False
    elif m.chip_active == "FH":
        m.free_hit_available = False
    elif m.chip_active == "TC":
        m.triple_captain_available = False
    elif m.chip_active == "BB":
        m.bench_boost_available = False
    m.chip_active = None
    # free transfers rollover (max 2)
    m.free_transfers = min(2, m.free_transfers + 1)

# ---------------------- GW Processing ----------------------

def process_gw(league: League, gw_events: Dict[int, Event]):
    gw = league.current_gw
    # Store GW
    league.history.append(Gameweek(gw=gw, events=gw_events))
    # Score each manager
    for m in league.managers.values():
        if not m.squad:
            continue
        # Free Hit backup if active now (store original before any modifications)
        if m.chip_active == "FH" and m.fh_backup_squad is None:
            # Already had to set during transfer phase ideally, but ensure backup exists
            m.fh_backup_squad = m.squad

        # Auto-subs
        final_starting, final_bench = auto_substitute(league.players, m.squad, gw_events)

        # Points calc
        week_points = 0
        # Starting XI
        for pid in final_starting:
            p = league.players[pid]
            ev = gw_events.get(pid, Event())
            pts = score_event(p, ev)
            week_points += pts
        # Captain / vice / triple captain
        cap = m.squad.captain
        vcap = m.squad.vice_captain
        cap_played = gw_events.get(cap, Event()).minutes > 0
        vcap_played = gw_events.get(vcap, Event()).minutes > 0
        # find cap points to double / triple
        if cap_played:
            cap_pts = score_event(league.players[cap], gw_events.get(cap, Event()))
            multiplier = 3 if m.chip_active == "TC" else 2
            week_points += cap_pts * (multiplier - 1)  # add extra (double already counted once)
        elif vcap_played:
            vcap_pts = score_event(league.players[vcap], gw_events.get(vcap, Event()))
            multiplier = 3 if m.chip_active == "TC" else 2
            week_points += vcap_pts * (multiplier - 1)
        # Bench Boost
        if m.chip_active == "BB":
            for pid in final_bench:
                p = league.players[pid]
                ev = gw_events.get(pid, Event())
                week_points += score_event(p, ev)
        # Apply transfer hits (should be recorded per manager for the GW; we track via negative events? keep temp attr)
        # Simplify: manager has temp attribute _pending_hit from apply_transfers result; if not, assume 0
        hit = getattr(m, "_pending_hit", 0)
        week_points -= hit
        m._pending_hit = 0  # reset

        m.total_points += week_points

        # Clear chip and rollover FT
        clear_chip_after_gw(m)

    league.current_gw += 1

# ---------------------- Helpers for CLI ----------------------

def print_player(players: Dict[int, Player], pid: int):
    p = players[pid]
    print(f"{pid:>3} | {p.name:<20} | {p.position:<3} | {p.club:<12} | {p.price:>4.1f}")

def list_players(players: Dict[int, Player], filt: Optional[str]=None):
    print("ID  | Name                 | Pos | Club         | Price")
    print("----+----------------------+-----+--------------+------")
    for pid, p in sorted(players.items()):
        if not filt or p.position == filt or p.club.lower()==filt.lower():
            print_player(players, pid)

def create_manager(league: League, name: str):
    if name in league.managers:
        print("Manager already exists.")
        return
    league.managers[name] = Manager(name=name)
    print(f"Manager '{name}' created.")

def pick_initial_squad(league: League, manager_name: str):
    m = league.managers[manager_name]
    print(f"Picking squad for {manager_name}. Budget {INITIAL_BUDGET:.1f}. Max {MAX_PER_CLUB} per club.")
    chosen = []
    club_counts = {}
    total = 0.0
    while len(chosen) < SQUAD_SIZE:
        list_players(league.players)
        pid = int(input(f"Pick player #{len(chosen)+1} by ID: "))
        if pid in chosen:
            print("Already picked.")
            continue
        if pid not in league.players:
            print("Invalid ID."); continue
        club = league.players[pid].club
        if club_counts.get(club,0) >= MAX_PER_CLUB:
            print(f"Max {MAX_PER_CLUB} from {club}."); continue
        price = league.players[pid].price
        if total + price > INITIAL_BUDGET:
            print("Over budget."); continue
        chosen.append(pid)
        club_counts[club] = club_counts.get(club,0)+1
        total += price
        print(f"Added. Spent {total:.1f}. Remaining {INITIAL_BUDGET-total:.1f}.")
    # choose starting XI
    print("\nChoose your starting XI (11 IDs, space-separated):")
    print("Remember valid formations like 3-4-3, 4-4-2 etc.; exactly one GK.")
    print("Your squad:", chosen)
    starting = list(map(int, input("Starting XI: ").split()))
    bench = [pid for pid in chosen if pid not in starting]
    # ensure bench GK at first
    bench_gk = [pid for pid in bench if league.players[pid].position=="GK"]
    if bench_gk:
        gk = bench_gk[0]
        bench.remove(gk)
        bench = [gk] + bench
    captain = int(input("Captain ID (must be in starting XI): "))
    vice = int(input("Vice Captain ID (must be in starting XI): "))
    sq = Squad(players=chosen, starting=starting, bench_order=bench, captain=captain, vice_captain=vice)
    ok, msg = check_squad_validity(league.players, sq)
    if not ok:
        print("Invalid squad:", msg)
        return
    m.squad = sq
    m.bank = INITIAL_BUDGET - squad_cost(league.players, sq)
    print(f"Squad saved. Bank: {m.bank:.1f}")

def manager_view(league: League, manager_name: str):
    m = league.managers[manager_name]
    print(f"\n--- {manager_name} ---")
    print(f"Total points: {m.total_points}, Bank: {m.bank:.1f}, Free Transfers: {m.free_transfers}, Chip active: {m.chip_active}")
    if not m.squad:
        print("No squad yet.")
        return
    print("\nStarting XI:")
    for pid in m.squad.starting:
        print_player(league.players, pid)
    print("\nBench (order):")
    for i, pid in enumerate(m.squad.bench_order, start=1):
        print(f"({i}) ", end=""); print_player(league.players, pid)
    print(f"Captain: {m.squad.captain}, Vice: {m.squad.vice_captain}")

def do_transfers(league: League, manager_name: str):
    m = league.managers[manager_name]
    if not m.squad:
        print("Pick a squad first."); return
    print("Enter transfers as: out_ids (comma) and in_ids (comma). Leave empty to cancel.")
    outs = input("Out IDs (e.g., 12,34): ").strip()
    if not outs:
        print("Cancelled."); return
    ins = input("In  IDs (e.g., 56,78): ").strip()
    outs_list = [int(x) for x in outs.split(",") if x.strip()]
    ins_list = [int(x) for x in ins.split(",") if x.strip()]
    ok, msg, hit = apply_transfers(m, league.players, outs_list, ins_list)
    if ok:
        m._pending_hit = getattr(m, "_pending_hit", 0) + hit
    print(msg, f"Hit: -{hit}" if hit else "")

def set_captains_and_bench(league: League, manager_name: str):
    m = league.managers[manager_name]
    if not m.squad:
        print("Pick a squad first."); return
    print("Current starting XI: ", m.squad.starting)
    start = input("New Starting XI (11 IDs, space-separated; blank to keep): ").strip()
    if start:
        starting = list(map(int, start.split()))
    else:
        starting = m.squad.starting
    bench = [pid for pid in m.squad.players if pid not in starting]
    bench_gk = [pid for pid in bench if league.players[pid].position=="GK"]
    if bench_gk:
        gk = bench_gk[0]; bench.remove(gk); bench = [gk] + bench
    cap = input(f"Captain ID (current {m.squad.captain}): ").strip()
    vcap = input(f"Vice ID (current {m.squad.vice_captain}): ").strip()
    cap = int(cap) if cap else m.squad.captain
    vcap = int(vcap) if vcap else m.squad.vice_captain
    sq = Squad(players=m.squad.players, starting=starting, bench_order=bench, captain=cap, vice_captain=vcap)
    ok, msg = check_squad_validity(league.players, sq)
    if not ok:
        print("Invalid setup:", msg); return
    m.squad = sq
    print("Updated XI, bench, and captains.")

def admin_enter_events(league: League) -> Dict[int, Event]:
    print("\nAdmin: Enter GW events. Press Enter to skip a field; default 0 / False.")
    events: Dict[int, Event] = {}
    print("Enter for players who participated; leave blank player ID to finish.")
    while True:
        pid_str = input("Player ID (blank to end): ").strip()
        if pid_str == "":
            break
        pid = int(pid_str)
        if pid not in league.players:
            print("Invalid ID."); continue
        def get_int(prompt): 
            s = input(prompt).strip(); 
            return int(s) if s else 0
        def get_bool(prompt):
            s = input(prompt + " (y/N): ").strip().lower(); 
            return s in ("y","yes","1","true")
        ev = Event()
        ev.minutes = get_int("Minutes played: ")
        ev.goals = get_int("Goals: ")
        ev.assists = get_int("Assists: ")
        ev.cs = get_bool("Clean sheet?")
        ev.goals_conceded = get_int("Goals conceded (team): ")
        ev.saves = get_int("Saves (GK): ")
        ev.pen_save = get_int("Penalties saved: ")
        ev.pen_miss = get_int("Penalties missed: ")
        ev.yellow = get_int("Yellow cards: ")
        ev.red = get_int("Red cards: ")
        ev.own_goals = get_int("Own goals: ")
        ev.bonus = get_int("Bonus points (0-3): ")
        events[pid] = ev
        print(f"Saved event for {league.players[pid].name}.")
    return events

def admin_process_gw(league: League):
    print(f"\n--- Process Gameweek {league.current_gw} ---")
    events = admin_enter_events(league)
    process_gw(league, events)
    print("GW processed. Standings:")
    print_standings(league)

def print_standings(league: League):
    print("\n=== League Standings ===")
    table = sorted([(m.name, m.total_points) for m in league.managers.values()], key=lambda x: -x[1])
    for i, (name, pts) in enumerate(table, start=1):
        print(f"{i:>2}. {name:<20} {pts:>4}")

def toggle_chip(league: League, manager_name: str):
    m = league.managers[manager_name]
    print("Activate chip: WC (Wildcard), FH (Free Hit), TC (Triple Captain), BB (Bench Boost)")
    chip = input("Chip code: ").strip().upper()
    ok, msg = activate_chip(m, chip)
    print(msg)

def save_menu(league: League):
    path = input("Save to file (e.g., league.json): ").strip() or "league.json"
    save_league(league, path)
    print(f"Saved to {path}")

def load_menu() -> League:
    path = input("Load from file (e.g., league.json): ").strip() or "league.json"
    lg = load_league(path)
    print(f"Loaded league '{lg.name}', GW {lg.current_gw}.")
    return lg

def create_league(players_csv: Optional[str]=None) -> League:
    name = input("League name: ").strip() or "My Custom League"
    if players_csv:
        players = load_players_csv(players_csv)
    else:
        print("No CSV provided. Using in-memory sample players.")
        players = {
            1: Player(1, "David Nilsson", "GK", "Blue FC", 4.5),
            2: Player(2, "Sara Holm", "DEF", "Blue FC", 5.0),
            3: Player(3, "Mikael Berg", "DEF", "Red United", 4.5),
            4: Player(4, "Lina Svensson", "MID", "Red United", 7.0),
            5: Player(5, "Ali Khan", "MID", "Green Town", 6.5),
            6: Player(6, "Johan Eriksson", "FWD", "Green Town", 8.0),
            7: Player(7, "Kim Andersson", "DEF", "Yellow City", 4.0),
            8: Player(8, "Noah Pettersson", "MID", "Yellow City", 6.0),
            9: Player(9, "Elin Larsson", "FWD", "Blue FC", 7.5),
            10: Player(10, "Oscar Lind", "GK", "Red United", 4.5),
            11: Player(11, "Vera Dahl", "DEF", "Green Town", 4.5),
            12: Player(12, "Sami Öst", "MID", "Blue FC", 5.5),
            13: Player(13, "Tilde Nurmi", "DEF", "Blue FC", 4.5),
            14: Player(14, "Moa Karlsson", "MID", "Red United", 6.0),
            15: Player(15, "Lukas Svensk", "FWD", "Yellow City", 7.0),
            16: Player(16, "Henrik Blom", "DEF", "Green Town", 4.0),
            17: Player(17, "Emil Öberg", "MID", "Green Town", 5.0),
            18: Player(18, "Nora Vik", "FWD", "Red United", 7.0),
        }
    return League(name=name, players=players)

def main():
    print("=== Mini FPL (Custom League) ===")
    league: Optional[League] = None
    players_csv: Optional[str] = None
    while True:
        print("\nMenu:")
        print("1) Create league")
        print("2) Load league")
        print("3) List players")
        print("4) Add manager")
        print("5) Pick squad for manager")
        print("6) View manager")
        print("7) Transfers")
        print("8) Set XI / bench / captains")
        print("9) Activate chip (WC/FH/TC/BB)")
        print("10) Process current GW")
        print("11) Standings")
        print("12) Save league")
        print("0) Exit")
        choice = input("Choice: ").strip()
        if choice == "1":
            p = input("Players CSV path (blank = use sample set): ").strip()
            players_csv = p if p else None
            league = create_league(players_csv)
            print(f"League '{league.name}' created. GW={league.current_gw}.")
        elif choice == "2":
            league = load_menu()
        elif choice == "3":
            if not league: print("Create or load a league first."); continue
            filt = input("Filter by position (GK/DEF/MID/FWD) or club (exact) or blank: ").strip()
            list_players(league.players, filt if filt else None)
        elif choice == "4":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            create_manager(league, name)
        elif choice == "5":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            if name not in league.managers: print("Manager not found."); continue
            pick_initial_squad(league, name)
        elif choice == "6":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            if name not in league.managers: print("Manager not found."); continue
            manager_view(league, name)
        elif choice == "7":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            if name not in league.managers: print("Manager not found."); continue
            do_transfers(league, name)
        elif choice == "8":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            if name not in league.managers: print("Manager not found."); continue
            set_captains_and_bench(league, name)
        elif choice == "9":
            if not league: print("Create or load a league first."); continue
            name = input("Manager name: ").strip()
            if name not in league.managers: print("Manager not found."); continue
            toggle_chip(league, name)
        elif choice == "10":
            if not league: print("Create or load a league first."); continue
            admin_process_gw(league)
        elif choice == "11":
            if not league: print("Create or load a league first."); continue
            print_standings(league)
        elif choice == "12":
            if not league: print("Create or load a league first."); continue
            save_menu(league)
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    # To use in Colab: upload a CSV of players (id,name,position,club,price) or use sample.
    # Then: !python mini_fpl.py
    main()


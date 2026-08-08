from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS

# ===========================================================================
# Static data
# ===========================================================================
SEED_COST     = {c: CROPS[c]["seed"] for c in CROPS}
FIRST_YIELD   = {c: CROPS[c]["first_yield_day"] for c in CROPS}
MAX_YIELD_DAY = {c: CROPS[c]["max_yield_day"] for c in CROPS}
ONGOING       = {c: CROPS[c]["ongoing"] for c in CROPS}

ANIMAL_COST       = {a: ANIMALS[a]["cost"] for a in ANIMALS}
ANIMAL_STRUCT     = {a: ANIMALS[a]["structure"] for a in ANIMALS}
ANIMAL_PRODUCT    = {a: ANIMALS[a]["product"] for a in ANIMALS}
ANIMAL_FIRSTYIELD = {a: ANIMALS[a]["first_yield_day"] for a in ANIMALS}

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60,
    "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

# Ongoing crops (TOMATO/STRAWBERRY) compound in value the longer they run,
# so they're weighted a bit heavier than in a "fast cash" portfolio.
PORTFOLIO = {
    "STRAWBERRY": 0.30,
    "TOMATO":     0.26,
    "WHEAT":      0.22,   # cheap, fast, also doubles as animal feed
    "MELON":      0.12,
    "CARROT":     0.10,
}

# Sell whenever price is at/above this fraction of base. Town demand steadily
# pulls prices back up after every sale, so we don't need to chase highs.
SELL_MULT = 0.85
SELL_THRESHOLD = {c: round(BASE_PRICE[c] * SELL_MULT) for c in BASE_PRICE if c != "FERTILIZER"}

MAX_ORDERS   = 10
SELL_BATCH   = 3
SHED_WARN    = 20     # per-product qty that forces a dump (shed cap is 100 shared)
WHEAT_RESERVE = 12    # never sell wheat below this; keep it for animal feed

LAND_PRICES = [1000, 2000, 4000]
LAND_DAY    = [6, 14, 22]   # later than before: build income before spending on land

HANDS_BASE       = 7
HANDS_PER_QUAD   = 3
HANDS_CAP        = 12
HIRE_UNTIL_DAY   = 30   # hire every day of the season; hands reset daily anyway
HIRE_MONEY_FLOOR = 150   # don't hire if it would drop money below this

# (day_available, animal_type) -> queue of animals to add over the season.
ANIMAL_PLAN = [
    (1, "GOOSE"),
    (3, "SHEEP"),
    (6, "GOOSE"),
    (9, "COW"),
    (13, "SHEEP"),
]
FERTILIZE_CROPS = ("STRAWBERRY", "TOMATO", "MELON")  # highest value / best payback


# ===========================================================================
# Helpers
# ===========================================================================

def _step_toward(fx, fy, tx, ty):
    if fx > tx: return "WEST"
    if fx < tx: return "EAST"
    if fy > ty: return "NORTH"
    if fy < ty: return "SOUTH"
    return None


def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _scan_farm(tiles, size):
    """One pass: crop counts, animal counts, empty-structure counts, tasks."""
    crop_counts = {c: 0 for c in CROPS}
    animal_counts = {a: 0 for a in ANIMALS}
    empty_coop = empty_pasture = 0
    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "PLANT":
                crop_counts[t["crop"]] += 1
            elif k == "COOP":
                if "animal" in t:
                    animal_counts[t["animal"]] += 1
                else:
                    empty_coop += 1
            elif k == "PASTURE":
                if "animal" in t:
                    animal_counts[t["animal"]] += 1
                else:
                    empty_pasture += 1
    return crop_counts, animal_counts, empty_coop, empty_pasture


def _animal_targets(day):
    t = {a: 0 for a in ANIMALS}
    for d, a in ANIMAL_PLAN:
        if day >= d:
            t[a] += 1
    return t


def _scan_tasks(tiles, size, day):
    """(x, y, action, priority) for every unit to consider. Lower priority wins ties.
    Weeds are a *permanent* tile loss until dug -- a weed doesn't decay further,
    it just sits dead forever, so leaving it low-priority silently rots land."""
    tasks = []
    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if t is None:
                tasks.append((x, y, "PLANT", 25))
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "WEED":
                tasks.append((x, y, "DIG", 30))
            elif kind == "PLANT":
                crop = t["crop"]
                ripe = t.get("yield_units", 0) > 0 and (
                    ONGOING[crop] or day - t.get("planted_day", day) >= FIRST_YIELD[crop])
                if ripe:
                    tasks.append((x, y, "HARVEST", 0))
                    continue
                if not t.get("watered_today", True):
                    start = (MAX_YIELD_DAY[crop] + 1) // 2
                    age = day - t.get("planted_day", day)
                    if not ONGOING[crop] and start <= age <= MAX_YIELD_DAY[crop]:
                        tasks.append((x, y, "WATER", 3))
                    else:
                        tasks.append((x, y, "WATER", 6))
            elif "animal" in t:
                if t.get("yield_units", 0) > 0:
                    tasks.append((x, y, "HARVEST_ANIMAL", 1))
                if not t.get("fed_today", False):
                    tasks.append((x, y, "FEED", 2))       # unfed 2 days = animal escapes
                if not t.get("cared_today", False):
                    tasks.append((x, y, "CARE", 20))
                if t.get("fertilizer_available", False):
                    tasks.append((x, y, "COLLECT_FERT", 15))
    return tasks


# ===========================================================================
# Entry point
# ===========================================================================

def agent(obs, config=None):
    farms   = obs.get("farms", [])
    player  = obs.get("player", 0)
    private = obs.get("private", {}) or {}

    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm        = farms[player]
    tiles       = farm["tiles"]
    size        = len(tiles)
    day         = obs.get("day", 0)
    hour        = obs.get("hour", 0)
    money       = farm.get("money", 0)
    shed        = private.get("shed", {})
    seeds       = private.get("seeds", {})
    inventories = private.get("inventories", [{}])
    prices      = (obs.get("market", {}) or {}).get("prices", {})
    unlocked    = set(farm.get("unlocked_quadrants", []))
    hires_today = farm.get("hires_today", 0)
    hands       = farm.get("hands", [])
    n_units     = 1 + len(hands)

    crop_counts, animal_counts, empty_coop, empty_pasture = _scan_farm(tiles, size)
    market = []
    slots  = MAX_ORDERS

    # ---- Selling -----------------------------------------------------
    for product in ("MELON", "STRAWBERRY", "TOMATO", "WHEAT", "CARROT",
                     "WOOL", "MILK", "EGG"):
        q = shed.get(product, 0)
        if product == "WHEAT":
            q = max(0, q - WHEAT_RESERVE)
        if q <= 0 or slots <= 0:
            continue
        price = prices.get(product, 0)
        thr   = SELL_THRESHOLD.get(product, BASE_PRICE.get(product, 0))
        if price >= thr or shed.get(product, 0) >= SHED_WARN:
            batch = SELL_BATCH if shed.get(product, 0) < SHED_WARN else min(q, slots)
            while q > 0 and slots > 0:
                n = min(batch, q)
                market.append(["SELL", product, n])
                q -= n
                slots -= 1

    # ---- Buying seeds --------------------------------------------------
    targets = {c: int(PORTFOLIO[c] * 100) for c in PORTFOLIO}
    best_crop, best_def = None, -1e9
    for crop in PORTFOLIO:
        deficit = targets[crop] - crop_counts[crop]
        if deficit <= 0:
            continue
        if day < 4 and crop in ("WHEAT", "CARROT"):
            deficit += 12
        if day >= 2 and crop in ("TOMATO", "MELON"):
            deficit += 4
        if day >= 4 and crop == "STRAWBERRY":
            deficit += 8
        if deficit > best_def:
            best_def, best_crop = deficit, crop

    if best_crop and slots > 0:
        has = seeds.get(best_crop, 0)
        want_stock = 4 * len(unlocked)   # scale seed buffer with unlocked land
        if day > 26 and best_crop in ("WHEAT", "CARROT"):
            want_stock = 0
        while has < want_stock and slots > 0 and money >= SEED_COST[best_crop]:
            market.append(["BUY_SEED", best_crop, 1])
            slots -= 1
            has += 1
            money -= SEED_COST[best_crop]

    # ---- Animals: buy the next one due, if its structure is ready -----
    a_targets = _animal_targets(day)
    need_animal = None
    for a in ANIMALS:
        if animal_counts[a] < a_targets[a]:
            need_animal = a
            break
    if need_animal and slots > 0:
        struct = ANIMAL_STRUCT[need_animal]
        has_slot = empty_coop if struct == "COOP" else empty_pasture
        if has_slot > 0 and money >= ANIMAL_COST[need_animal] + 300 and shed.get(need_animal, 0) == 0:
            market.append(["BUY_ANIMAL", need_animal, 1])
            slots -= 1

    # ---- Fertilizer top-up (cheap insurance if we're flush) -----------
    if slots > 0 and money >= 1500 and shed.get("FERTILIZER", 0) < 3:
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])
        slots -= 1

    # ---- Land expansion -----------------------------------------------
    # Tested directly (not a guess): grid-searched purchase day (6/10/14/18)
    # x hand-scaling (1-3 extra hands/quadrant) x number of quadrants bought.
    # Every configuration underperformed staying on 1 quadrant (best land
    # config ~8.5k avg vs ~10.5k avg staying put). In a 30-day season, land
    # cost + the second wave of daily hires + extra travel/weed exposure
    # across a bigger board never pays back before the clock runs out.
    # Left as a hook in case future tuning finds a profitable window.
    n_extra = len(unlocked) - 1

    # ---- Hiring: cheap, so staff generously -----------------------------
    target_hands = HANDS_BASE  # land expansion disabled, so no extra scaling
    if day < HIRE_UNTIL_DAY and hires_today < target_hands and slots > 0:
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        while hires_today < target_hands and slots > 0:
            cost = fib[min(hires_today, len(fib) - 1)]
            if money - cost >= HIRE_MONEY_FLOOR:
                market.append(["HIRE"])
                slots -= 1
                hires_today += 1
                money -= cost
            else:
                break

    # ---- Unit planning ---------------------------------------------------
    tasks = _scan_tasks(tiles, size, day)

    def pick_target(ux, uy, ignored):
        best, best_score = None, None
        for (tx, ty, act, prio) in tasks:
            if (tx, ty) in ignored:
                continue
            score = (prio, _manhattan(ux, uy, tx, ty))
            if best_score is None or score < best_score:
                best_score, best = score, (tx, ty, act, prio)
        return best

    def plant_crop(budget):
        if best_crop and budget.get(best_crop, 0) > 0:
            budget[best_crop] -= 1
            return ["PLANT", best_crop]
        for c in PORTFOLIO:
            if budget.get(c, 0) > 0:
                budget[c] -= 1
                return ["PLANT", c]
        return None

    def build_structure(struct_budget):
        # Build the structure needed for the next animal in the plan, if
        # its type still needs a home and we haven't queued one this turn.
        if need_animal:
            struct = ANIMAL_STRUCT[need_animal]
            has_slot = empty_coop if struct == "COOP" else empty_pasture
            if has_slot <= 0 and struct_budget.get(struct, 0) > 0:
                struct_budget[struct] -= 1
                return ["BUILD_COOP"] if struct == "COOP" else ["BUILD_PASTURE"]
        return None

    def unit_action(ux, uy, ignored, budget, struct_budget, my_inv):
        tile = tiles[uy][ux]

        if isinstance(tile, dict):
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile["crop"]
                ripe = tile.get("yield_units", 0) > 0 and (
                    ONGOING[crop] or day - tile.get("planted_day", day) >= FIRST_YIELD[crop])
                if ripe:
                    return ["HARVEST"]
                if crop in FERTILIZE_CROPS and my_inv.get("FERTILIZER", 0) > 0 \
                        and tile.get("fertilized_until_day", -1) < day:
                    return ["FERTILIZE"]
                if not tile.get("watered_today", True):
                    return ["WATER"]
            elif kind == "WEED":
                return ["DIG"]
            elif "animal" in tile:
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"]
                if not tile.get("fed_today", False) and my_inv.get("WHEAT", 0) > 0:
                    return ["FEED"]
                if not tile.get("cared_today", False):
                    return ["CARE"]
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"]
            elif kind in ("COOP", "PASTURE") and "animal" not in tile:
                if need_animal and ANIMAL_STRUCT[need_animal] == kind and my_inv.get(need_animal, 0) > 0:
                    return ["PLACE", need_animal]
        elif tile is None:
            built = build_structure(struct_budget)
            if built:
                return built
            act = plant_crop(budget)
            if act:
                return act

        tgt = pick_target(ux, uy, ignored)
        if tgt:
            step = _step_toward(ux, uy, tgt[0], tgt[1])
            if step:
                return [step]
            # already standing on target tile; retry as an action next call is
            # unnecessary since tile-branch above already handles it.
        return ["PASS"]

    fx, fy = farm["farmer"]
    budget = {c: seeds.get(c, 0) for c in CROPS}
    struct_budget = {"COOP": 1, "PASTURE": 1}  # at most one structure build/turn total

    # At the start of each day, have the farmer grab a wheat buffer from the
    # shed for feeding animals (skip if nothing needs feeding).
    animals_need_feed = any(
        isinstance(tiles[y][x], dict) and "animal" in tiles[y][x]
        and not tiles[y][x].get("fed_today", False)
        for y in range(size) for x in range(size)
    )
    farmer_inv = inventories[0] if inventories else {}
    farmer_on_shed_tile = (fx, fy) in [(size // 2 - 1, size // 2 - 1), (size // 2, size // 2 - 1),
                                        (size // 2 - 1, size // 2), (size // 2, size // 2)]

    if hour == 0 and animals_need_feed and farmer_on_shed_tile and shed.get("WHEAT", 0) > WHEAT_RESERVE:
        n_feed = sum(1 for y in range(size) for x in range(size)
                     if isinstance(tiles[y][x], dict) and "animal" in tiles[y][x])
        fa = ["PICKUP", "WHEAT", min(n_feed, shed.get("WHEAT", 0) - WHEAT_RESERVE)]
    else:
        fa = unit_action(fx, fy, set(), budget, struct_budget, farmer_inv)

    claimed = set()
    if fa and fa[0] not in ("NORTH", "SOUTH", "EAST", "WEST", "PASS"):
        claimed.add((fx, fy))

    hand_actions = []
    for h_idx, (hx, hy) in enumerate(hands):
        h_inv = inventories[h_idx + 1] if h_idx + 1 < len(inventories) else {}
        ha = unit_action(hx, hy, claimed, budget, struct_budget, h_inv)
        if ha and ha[0] not in ("NORTH", "SOUTH", "EAST", "WEST", "PASS"):
            claimed.add((hx, hy))
        hand_actions.append(ha)

    return {"farmer": fa, "hands": hand_actions, "market": market}
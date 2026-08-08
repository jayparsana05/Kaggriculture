"""Kaggriculture agent v5 — economic planner.

Architecture (per task spec):
    analyze_farm / analyze_market / analyze_opponent
    estimate_crop_profit -> crop_plan
    sell_plan (marginal market-price model + town demand model)
    invest_plan (land / animals / fertilizer / hands)
    build_tasks -> schedule_units (deadline-aware, cluster-friendly scheduler)

Core principles:
1.  Watering is absolute: plants with consecutive_unwatered >= 1 outrank
    everything; weeds are permanent income loss.
2.  Selling uses the real marginal price curve: never assume SELL 10 pays the
    current price for every unit. Premium goods (strawberry/melon/milk/wool)
    collapse toward $1 on gluts - pace sales to what the town absorbs.
3.  Crop choice is economic: expected profit = units * expected price
    (projected through town demand) - seed - fertilizer - labor, capped by
    market absorption and opponent production.
4.  Cash discipline: never let money starve. Seeds before hands before
    animals before land, all gated by a cash buffer.
5.  Endgame: CASH > INVENTORY. Liquidate everything from day 27 on.
"""
import math
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    SHOPS,
)

# ============================================================================
# Static data
# ============================================================================
SEED_COST     = {c: CROPS[c]["seed"] for c in CROPS}
FIRST_YIELD   = {c: CROPS[c]["first_yield_day"] for c in CROPS}
MAX_YIELD_DAY = {c: CROPS[c]["max_yield_day"] for c in CROPS}
ONGOING       = {c: CROPS[c]["ongoing"] for c in CROPS}
INTERVAL      = {c: CROPS[c]["interval"] for c in CROPS}
MAX_YIELD     = {c: CROPS[c]["max_yield"] for c in CROPS}

ANIMAL_COST    = {a: ANIMALS[a]["cost"] for a in ANIMALS}
ANIMAL_STRUCT  = {a: ANIMALS[a]["structure"] for a in ANIMALS}
ANIMAL_PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMALS}
ANIMAL_FIRST   = {a: ANIMALS[a]["first_yield_day"] for a in ANIMALS}
ANIMAL_INTERVAL = {a: ANIMALS[a]["interval"] for a in ANIMALS}
ANIMAL_HELD    = {a: ANIMALS[a]["max_held"] for a in ANIMALS}

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}
CROPS_LIST = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_LIST = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
SELL_ORDER = ["MELON", "STRAWBERRY", "WOOL", "MILK", "TOMATO", "EGG",
              "WHEAT", "CARROT", "FERTILIZER"]

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# ============================================================================
# Tuning knobs
# ============================================================================
MAX_ORDERS    = 10
LAST_DAY      = 29                 # everything must be harvested by day 28
                                   # (day-29 harvests can never be sold) and
                                   # sold by day 29 hour 22.
LAST_HARVEST  = 28

CASH_FLOOR        = 80
CASH_FLOW_THRESH  = 500
ANIMALS_ENABLED   = True
ANIMAL_CAPS       = {"GOOSE": 10, "COW": 4, "SHEEP": 2}
FERT_PRICE_MIN    = 62
FERT_USE_MAX      = 8

HARD_CAPS = {"WHEAT": 60, "CARROT": 30, "TOMATO": 10,
             "STRAWBERRY": 14, "MELON": 16}

ENDGAME_DAY   = 27
SHED_FORCE    = 86
ACTIONS_PER_HAND = 14
HANDS_MAX     = 6
HAND_VALUE_MIN  = 6
HIRE_HOUR_CUT = 12

LAND_DAYS     = [4, 8, 14]
LAND_CASH     = [1000, 2000, 4000]
LAND_LAST_DAY = 24

LABOR_VALUE   = 3
GLUT_FRACTION = 0.80

# Opponent production model (explicit probabilities, no arbitrary 1.4)
OPP_HARVEST      = 1.05   # expected crop output that reaches the market:
                          # real opponents harvest near-fully (1.0) with a
                          # modest uncertainty band (0.85-1.15), not 1.4
OPP_CARE         = 0.70   # opponent care probability on production days
OPP_SURVIVAL     = 0.95   # placed animals survive to keep producing
OPP_SELL_FRACTION = 0.80  # fraction of opponent production that reaches the
                          # market (they undersell; we still price it in)
OPP_FERT_RATE    = 0.60   # expected fertilizer collected per opponent animal/day
OPP_ANIMAL_FACTOR = (1.0 + OPP_CARE) * OPP_SURVIVAL * OPP_SELL_FRACTION

TRAVEL_COST  = 8
REGION_BIAS  = 3

# Task priority bands (higher = more urgent); value bonuses add at most +700.
# Animal infrastructure outranks routine plant care: one built coop + placed
# goose pays ~$150/day, far more than any single plant-watering action.
P_PICKUP_WHEAT = 9850
P_PICKUP_FERT  = 9750
P_FEED_CRIT    = 9700   # consecutive_unfed >= 1: must feed today
P_WATER_CRIT   = 9600   # consecutive_unwatered >= 1: must water today
P_FEED         = 9550   # feeding preserves a permanent animal asset
P_HARVEST      = 9450   # ripe crop cash beats one-time building chores
P_BUILD        = 9400
P_PLACE        = 9350
P_PICKUP_ANIMAL = 9300
P_COLLECT      = 9200
P_CARE         = 9150
P_WATER_WINDOW = 8900   # bonus-window watering of one-time crops
P_FERTILIZE    = 8800
P_WATER        = 8600
P_DIG_SPENT    = 7800
P_PLANT        = 7000
P_DIG_WEED     = 6400

_PRICE_HIST = {}        # product -> [(day, price), ...] recorded at hour 0
PREMIUM = {"STRAWBERRY", "MELON", "MILK", "WOOL"}   # glut-sensitive products


# ============================================================================
# Market model
# ============================================================================
def _shape(func, x):
    import math
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    return x


def _params(market):
    return market.get("params") or MARKET_PARAMS


def price_at(item, inventory, params):
    """Exact replica of the env's market_price."""
    p = params[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T)
        price = base + amp * _shape(f, I0 - inventory)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T)
        price = base - amp * _shape(f, inventory - I0)
    return max(1, int(round(price)))


def batch_avg(item, n, inventory, params):
    """Average price for selling n units right now (sequential pricing;
    floor-$1 sales do not add to market inventory)."""
    if n <= 0:
        return price_at(item, inventory, params)
    total = 0
    inv = inventory
    for _ in range(n):
        p = price_at(item, inv, params)
        total += p
        if p > 1:
            inv += 1
    return total / n


def glut_tolerance(item, params, target_frac=GLUT_FRACTION):
    """Surplus units above I0 the market absorbs before price < frac*base."""
    base = params[item]["base"]
    target = max(1, int(round(base * target_frac)))
    d = 0
    while d < 2000 and price_at(item, 10000 + d, params) >= target:
        d += 1
    return d


def town_rate(day, product, town):
    """Expected town consumption of `product` on `day` (units/day):
    town center (2/4/8) + unlocked shops (6/day each, 12 for single-product
    shops) + expected future shop unlocks."""
    mult = 8 if day >= 20 else (4 if day >= 10 else 2)
    rate = float(mult)
    unlocked = set(town.get("unlocked_shops", []) or [])
    for s in unlocked:
        if product in SHOPS[s]:
            rate += 12.0 if len(SHOPS[s]) == 1 else 6.0
    expected_total = min(8, (day + 1) // 3)
    k_future = max(0, expected_total - len(unlocked))
    remaining = [s for s in SHOPS if s not in unlocked]
    if k_future and remaining:
        # Future unlocks are speculative: weight them at half confidence so
        # known (unlocked) demand dominates the projection.
        for s in remaining:
            if product in SHOPS[s]:
                rate += (12.0 if len(SHOPS[s]) == 1 else 6.0) \
                    * k_future / len(remaining) * 0.5
    return rate


def drain_between(day, product, days, town):
    total = 0.0
    for d in range(day, min(day + days, LAST_DAY)):
        total += town_rate(d, product, town)
    return total


def project_price(item, inv, day, days, town, opp_rate, params, my_rate=0.0):
    """Expected price `days` from now given town drain and total supply
    pressure (opponent sales + our own committed future production)."""
    drain = drain_between(day, item, days, town)
    inv_f = max(1.0, inv - drain + (opp_rate + my_rate) * days)
    return price_at(item, inv_f, params)


def price_trend(item, day):
    """Per-day price velocity from the recorded history (last 3 points).
    Returns None when there is not enough data."""
    hist = _PRICE_HIST.get(item)
    if not hist or len(hist) < 2:
        return None
    pts = [p for (d, p) in hist if d < day]
    if len(pts) < 2:
        return None
    span = max(1, hist[-1][0] - hist[-2][0])
    return (hist[-1][1] - hist[-2][1]) / span


def price_state(item, day, prices, params):
    """Classify a product's market into velocity x level:
    ('RISING'|'FALLING'|'STABLE') x ('CHEAP'|'NORMAL'|'EXPENSIVE').
    Velocity comes from the recorded day-0 price series (both above and
    below base price); level from the current price vs base."""
    vel = price_trend(item, day) or 0.0
    if vel >= 1.5:
        v = "RISING"
    elif vel <= -1.5:
        v = "FALLING"
    else:
        v = "STABLE"
    base = params[item]["base"]
    price = prices.get(item, base)
    if price <= 0.85 * base:
        lv = "CHEAP"
    elif price >= 1.15 * base:
        lv = "EXPENSIVE"
    else:
        lv = "NORMAL"
    return v, lv


def expected_cash_value(item, qty, horizon_days, day, prices, inv, town,
                        opp_rate, params, my_rate=0.0):
    """Expected cash from holding `qty` units and selling over the next
    `horizon_days` at the projected (drained) price, vs the marginal sell-out
    now.  Returns (hold_value, sell_now_value) per unit."""
    est = project_price(item, inv.get(item, params[item]["I0"]), day,
                        horizon_days, town, opp_rate, params, my_rate)
    return est, batch_avg(item, qty, inv.get(item, params[item]["I0"]), params)


# ============================================================================
# Farm analysis
# ============================================================================
def _productions_remaining(crop, planted_day, day):
    n = 0
    for k in range(MAX_YIELD[crop]):
        nd = planted_day + FIRST_YIELD[crop] + k * INTERVAL[crop]
        if nd > day and nd <= LAST_HARVEST:
            n += 1
    return n


def _animal_prods_left(animal, placed_day, day):
    """Production ticks still coming from a placed animal (fed+cared)."""
    n = 0
    k = 0
    while True:
        nd = placed_day + ANIMAL_FIRST[animal] + k * ANIMAL_INTERVAL[animal]
        if nd > LAST_HARVEST:
            break
        if nd > day:
            n += 1
        k += 1
    return n


# ============================================================================
# Canonical animal production model (single source of truth)
# ============================================================================
def animal_base_rate(animal):
    """Biological base production of `animal` in units/day (no CARE):
    GOOSE 1.0, COW 0.5, SHEEP 0.333.  The env grants one unit per production
    tick; ticks/day = 1 / interval."""
    return 1.0 / ANIMAL_INTERVAL[animal]


def care_probability(farm_state, day):
    """Expected probability that a fed animal is CAREd on a production day.

    CARE competes with watering, harvesting, feeding, etc. and is never
    guaranteed: estimate it from current worker saturation.  A farm with
    spare labor cares reliably; an overloaded farm skips care.
    """
    if not farm_state.get("n_animals"):
        return 1.0
    workload = 5.0 + farm_state["n_animals"] * 3.0
    for (x, y, t) in farm_state.get("plants", []):
        workload += _tile_demand(t, day)
    capacity = max(1, farm_state.get("n_units", 1)) * ACTIONS_PER_HAND
    spare = max(0.0, min(1.0, 1.0 - workload / capacity))
    return min(0.9, max(0.3, 0.3 + 0.6 * spare))


def animal_expected_units(animal, placed_day, day, farm_state):
    """Expected total units of `animal`'s product still to come, including
    the probabilistic CARE bonus: per tick the env pays 1 base unit plus
    one bonus unit iff CAREd, so expected units/tick = 1 + care_probability."""
    ticks = _animal_prods_left(animal, placed_day, day)
    return ticks * (1.0 + care_probability(farm_state, day))


def future_supply_schedule(farm_state, day, horizon=10):
    """Expected per-day additions of each product to OUR inventory over the
    next `horizon` days (production happens on specific days, not uniformly).

    Returns {product: [units_on_day_offset_0, ... offset_H]}.  Used for
    sell pacing and the lightweight forward projection.
    """
    H = max(1, horizon)
    out = {p: [0.0] * (H + 1) for p in PRODUCTS}
    last = min(LAST_HARVEST, day + H)
    cp = care_probability(farm_state, day)
    for (x, y, t) in farm_state["plants"]:
        c = t["crop"]
        if ONGOING[c]:
            for k in range(MAX_YIELD[c]):
                nd = t["planted_day"] + FIRST_YIELD[c] + k * INTERVAL[c]
                if day < nd <= last:
                    out[c][nd - day] += 1.0
        else:
            rem = max(0, t.get("yield_units", 0))
            if rem > 0:
                out[c][1] += rem                    # already grown: sell soon
            start = (MAX_YIELD_DAY[c] + 1) // 2
            for wd in range(max(start, day + 1), min(MAX_YIELD_DAY[c], last) + 1):
                out[c][wd - day] += 0.95            # ~1 bonus unit per window day
    for (x, y, t) in farm_state["animal_tiles"]:
        a = t["animal"]
        p = ANIMAL_PRODUCT[a]
        k = 0
        while True:
            nd = t.get("placed_day", day) + ANIMAL_FIRST[a] + k * ANIMAL_INTERVAL[a]
            if nd > last:
                break
            if nd > day:
                out[p][nd - day] += 1.0 + cp
            k += 1
    return out


def committed_units(farm_state, day, product):
    """Total units of `product` our farm is already committed to produce
    (existing plants' remaining yield + placed animals' remaining output)."""
    total = 0.0
    for (x, y, t) in farm_state["plants"]:
        if t["crop"] != product:
            continue
        if ONGOING[product]:
            total += _productions_remaining(product, t["planted_day"], day)
        else:
            rem = max(0, t.get("yield_units", 0))
            start = (MAX_YIELD_DAY[product] + 1) // 2
            for wd in range(max(start, day + 1), MAX_YIELD_DAY[product] + 1):
                if wd <= LAST_HARVEST:
                    rem += 1
            total += rem
    for (x, y, t) in farm_state["animal_tiles"]:
        a = t["animal"]
        if ANIMAL_PRODUCT[a] == product:
            total += animal_expected_units(a, t.get("placed_day", day), day,
                                           farm_state)
    return total


def my_supply_rates(farm_state, day):
    """Units/day of each product our committed assets will add to the market."""
    days_left = max(1, LAST_DAY - day)
    rates = {}
    for p in PRODUCTS:
        rates[p] = committed_units(farm_state, day, p) / days_left
    return rates


def forward_bank(day, horizon, prices, inv, town, opp, params, farm_state,
                 shed=None):
    """Projected bank cash from selling all near-term committed supply over
    the next `horizon` days, priced through the market projection (town
    drain, opponent sales, our own sales pressure).  A lightweight
    deterministic forward pass - used only for big decisions (land), never
    per action."""
    days_left = max(1, LAST_DAY - day)
    H = min(max(1, horizon), days_left)
    sched = future_supply_schedule(farm_state, day, H)
    bank = 0.0
    opp_r = opp["daily_rate"]
    for p in PRODUCTS:
        if p == "FERTILIZER":
            continue
        ours = (shed or {}).get(p, 0) + sum(sched.get(p, [0.0] * (H + 1)))
        if ours <= 0:
            continue
        opp_sell = opp_r.get(p, 0.0) * OPP_SELL_FRACTION
        est_p = project_price(p, inv.get(p, params[p]["I0"]), day, H,
                              town, opp_sell, params, ours / max(1, H))
        bank += ours * est_p
    return bank


def analyze_farm(farm, day):
    tiles = farm["tiles"]
    size = len(tiles)
    out = {
        "size": size,
        "crop_counts": {c: 0 for c in CROPS_LIST},
        "animal_counts": {a: 0 for a in ANIMALS_LIST},
        "empty": 0, "weeds": 0, "empty_coop": 0, "empty_pasture": 0,
        "spent": [],          # (x, y, crop) finished ongoing crops
        "unwatered": 0, "unwatered_at_risk": 0,
        "unfed": 0, "unfed_at_risk": 0,
        "plants": [],         # (x, y, tile)
        "animal_tiles": [],   # (x, y, tile)
        "n_animals": 0,
    }
    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if t is None:
                out["empty"] += 1
            elif not isinstance(t, dict):
                continue
            elif t.get("kind") == "WEED":
                out["weeds"] += 1
            elif t.get("kind") == "PLANT":
                crop = t["crop"]
                out["crop_counts"][crop] += 1
                out["plants"].append((x, y, t))
                if not t.get("watered_today", True):
                    out["unwatered"] += 1
                    if t.get("consecutive_unwatered", 0) >= 1:
                        out["unwatered_at_risk"] += 1
                if ONGOING[crop] and _productions_remaining(crop, t["planted_day"], day) == 0:
                    out["spent"].append((x, y, crop))
            elif "animal" in t:
                out["animal_counts"][t["animal"]] += 1
                out["n_animals"] += 1
                out["animal_tiles"].append((x, y, t))
                if not t.get("fed_today", False):
                    out["unfed"] += 1
                    if t.get("consecutive_unfed", 0) >= 1:
                        out["unfed_at_risk"] += 1
            elif t.get("kind") == "COOP":
                out["empty_coop"] += 1
            elif t.get("kind") == "PASTURE":
                out["empty_pasture"] += 1
    return out


def analyze_opponent(farms, player, day):
    """Estimate the opponent's expected MARKET SALES per day of each product.

    Biological production is converted to sales with explicit probabilities
    (harvest, care, survival, selling) instead of an arbitrary 1.4 boost:
    opponents do not harvest/water/sell everything perfectly.
    """
    opp_farm = farms[1 - player]
    tiles = opp_farm["tiles"]
    size = len(tiles)
    out = {
        "crop_counts": {c: 0 for c in CROPS_LIST},
        "animal_counts": {a: 0 for a in ANIMALS_LIST},
        "n_unlocked": len(opp_farm.get("unlocked_quadrants", []) or []),
        "n_hands": len(opp_farm.get("hands", []) or []),
        "daily_rate": {p: 0.0 for p in PRODUCTS},
    }
    days_left = max(1, LAST_DAY - day)
    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "PLANT":
                crop = t["crop"]
                out["crop_counts"][crop] += 1
                if ONGOING[crop]:
                    rem = _productions_remaining(crop, t["planted_day"], day)
                    if rem > 0:
                        # expected biological output x harvest+sell probability,
                        # spread uniformly over the remaining days
                        out["daily_rate"][crop] += rem * OPP_HARVEST / days_left
                elif t.get("yield_units", 0) > 0:
                    out["daily_rate"][crop] += \
                        max(1.0, t["yield_units"] * OPP_HARVEST) / days_left
            elif "animal" in t:
                a = t["animal"]
                out["animal_counts"][a] += 1
                p = ANIMAL_PRODUCT[a]
                # per-day expected sales: base rate x (1 + expected CARE bonus)
                # x survival; the old model's (1 + 1/interval)/days_left
                # undercounted a goose's whole-season supply to ~2 units.
                out["daily_rate"][p] += animal_base_rate(a) * OPP_ANIMAL_FACTOR
                out["daily_rate"]["FERTILIZER"] += OPP_FERT_RATE
    return out


# ============================================================================
# Crop economics
# ============================================================================
def units_estimate(crop, plant_day, with_fert):
    """Expected total units a plant planted today will produce by end of
    season if watering is maintained and fertilizer applied when it pays."""
    if ONGOING[crop]:
        n = 0
        for k in range(MAX_YIELD[crop]):
            nd = plant_day + FIRST_YIELD[crop] + k * INTERVAL[crop]
            if nd <= LAST_HARVEST:
                n += 1
        if with_fert:
            n += min(n, 3)      # ~3 of 4 productions doubled by 2 fertilizer
        return n
    start = (MAX_YIELD_DAY[crop] + 1) // 2
    water_days = max(0, MAX_YIELD_DAY[crop] - max(start, 1) + 1)
    bonus = 2 if with_fert else 1
    return min(MAX_YIELD[crop], 1 + water_days * bonus)


def can_finish(crop, day):
    if ONGOING[crop]:
        return day + FIRST_YIELD[crop] <= LAST_HARVEST
    return day + MAX_YIELD_DAY[crop] <= LAST_HARVEST


def _fert_marginal(crop, day, prices, params, market_state=None):
    """Dollar value of fertilizing one plant of `crop` vs selling the fert.

    Uses the same projected harvest price as crop planning (cached in
    market_state["harvest_price"] by crop_plan), never today's spot price:
    the bonus units land at future production days, so they must be valued
    at the projected price.
    """
    if crop not in ("STRAWBERRY", "TOMATO", "WHEAT"):
        return -1.0
    if market_state and market_state.get("harvest_price"):
        est_p = market_state["harvest_price"].get(crop)
    else:
        est_p = None
    if est_p is None:
        est_p = prices.get(crop, BASE_PRICE[crop])
    if crop == "STRAWBERRY":
        added = 2
    elif crop == "TOMATO":
        added = 3
    else:
        added = 2 if MAX_YIELD[crop] - units_estimate(crop, day, False) >= 2 else 0
    return added * est_p - prices.get("FERTILIZER", BASE_PRICE["FERTILIZER"])


def wave_profit(crop, day, prices, inv, params, town, opp, my_rate=0.0,
                market_state=None):
    """Expected profit of one wave of `crop` planted today (per tile)."""
    if not can_finish(crop, day):
        return -1e9
    with_fert = _fert_marginal(crop, day, prices, params, market_state) > 0
    units = units_estimate(crop, day, with_fert)
    if units <= 0:
        return -1e9
    horizon = FIRST_YIELD[crop] if ONGOING[crop] else MAX_YIELD_DAY[crop]
    est_p = project_price(crop, inv.get(crop, params[crop]["I0"]), day,
                          horizon, town, opp.get("daily_rate", {}).get(crop, 0.0),
                          params, my_rate)
    if crop == "WHEAT":
        # Wheat is both a market good and animal feed.  Its effective value
        # is the better of (a) the projected sale price and (b) the feed
        # replacement value = what we would pay to BUY wheat on the market
        # today (our committed wheat mostly feeds animals, not the market).
        est_p = max(est_p, prices.get("WHEAT", BASE_PRICE["WHEAT"]))
    revenue = units * est_p
    fert_cost = 0.0
    if with_fert:
        n_fert = 3 if crop == "STRAWBERRY" else 1
        fert_cost = n_fert * prices.get("FERTILIZER", BASE_PRICE["FERTILIZER"])
    labor = (FIRST_YIELD[crop] if ONGOING[crop] else MAX_YIELD_DAY[crop]) + 4
    return revenue - SEED_COST[crop] - fert_cost - labor * LABOR_VALUE


def _waves_left(crop, day):
    if ONGOING[crop]:
        return max(1, (LAST_DAY - day) // (FIRST_YIELD[crop] + INTERVAL[crop] * 2))
    return max(1, (LAST_DAY - day) // (MAX_YIELD_DAY[crop] + 2))


def allowance_tiles(crop, day, prices, inv, params, town, opp, my_rate=0.0):
    """Max tiles of `crop` we may plant without self-crashing the market.

    safe future capacity = tolerance + remaining town drain
                          - opponent supply - our committed supply
                          - current market surplus above I0
    """
    I0 = params[crop]["I0"]
    tol = glut_tolerance(crop, params)
    drain_rem = drain_between(day, crop, LAST_DAY - day, town)
    opp_est = opp["daily_rate"].get(crop, 0.0) * (LAST_DAY - day)
    my_units = my_rate * (LAST_DAY - day)
    surplus = max(0, inv.get(crop, I0) - I0)
    safe_units = tol + drain_rem - opp_est - my_units - surplus
    per_tile = max(1, units_estimate(crop, day, False) * _waves_left(crop, day))
    return max(1.0, safe_units / per_tile)


def _tile_demand(t, day):
    """Estimated actions/day needed to keep `t` (a plant tile) productive:
    ongoing crops demand water+harvest while productions remain; one-time
    crops demand water only inside their bonus window and then just a
    harvest."""
    c = t["crop"]
    if ONGOING[c]:
        if _productions_remaining(c, t["planted_day"], day) > 0:
            return 2.2
        return 0.3                     # spent: just a DIG
    if day >= MAX_YIELD_DAY[c]:
        return 0.8                     # matured: harvest only
    return 2.2                         # growing phase


def crop_plan(farm_state, market_state, opp, day, prices, inv, params, money,
              land_extra=0):
    tiles_avail = farm_state["empty"] + farm_state["weeds"] + len(farm_state["spent"]) \
        + land_extra
    market_state.setdefault("harvest_price",
                            {c: prices.get(c, BASE_PRICE[c]) for c in CROPS_LIST})
    if tiles_avail <= 0:
        return {"crops": {}, "ranked": [], "profit": {}, "fert_reserve": 0,
                "tiles_avail": 0}

    # Labor capacity: estimate the farm's real action demand (watering,
    # harvesting, planting, fertilizing, feeding, caring, movement) and gate
    # new plantings on the spare capacity of the hired crew.  Count the hands
    # we are about to hire today (hired hands work from tomorrow).
    n_units = farm_state.get("n_units", 1)
    if money >= 2500:
        planned = 6
    elif money >= 1200:
        planned = 5
    elif money >= 600:
        planned = 4
    elif money >= 200:
        planned = 3
    else:
        planned = 2
    n_units = min(HANDS_MAX + 1, max(n_units, 1 + planned))
    capacity = n_units * ACTIONS_PER_HAND
    demand_now = 5.0
    for (x, y, t) in farm_state["plants"]:
        demand_now += _tile_demand(t, day)
    demand_now += farm_state["n_animals"] * 3.0
    labor_spare = capacity - demand_now
    # Mandatory-task reservation: feeding animals and re-watering at-risk
    # plants are hard requirements.  Reserve their action budget BEFORE any
    # optional planting is allowed; planting must shrink before animals go
    # unfed.  A small tolerance remains for one-time crops that soon leave
    # their water window.
    mandatory_reserve = farm_state["n_animals"] * 2.0 \
        + farm_state["unwatered_at_risk"] * 1.5
    tiles_avail = min(tiles_avail,
                      max(0, int((labor_spare - mandatory_reserve + 8.0) / 2.2)))

    # Seasonality: crops whose payoff window has passed are never worth
    # planting this late.
    late_skip = {"MELON": 9, "TOMATO": 20, "STRAWBERRY": 22,
                 "CARROT": 24, "WHEAT": 27}

    my_rate = my_supply_rates(farm_state, day)

    # Cache projected harvest prices for fertilizer decisions so crop
    # planning, fertilizer, and selling all value a unit the same way.
    hp = market_state["harvest_price"]
    for c in CROPS_LIST:
        h = FIRST_YIELD[c] if ONGOING[c] else MAX_YIELD_DAY[c]
        hp[c] = project_price(c, inv.get(c, params[c]["I0"]), day, h,
                              market_state["town"],
                              opp.get("daily_rate", {}).get(c, 0.0) * OPP_SELL_FRACTION,
                              params, my_rate.get(c, 0.0))

    profits = {}
    horizon = {}
    for c in CROPS_LIST:
        if day > late_skip.get(c, 30):
            continue
        # Dead market: never add supply to a crashed price, except the feed
        # staple wheat (its floor price is all we ever get anyway).
        if c != "WHEAT" and prices.get(c, BASE_PRICE[c]) < 0.6 * BASE_PRICE[c]:
            continue
        profits[c] = wave_profit(c, day, prices, inv, params,
                                 market_state["town"], opp, my_rate.get(c, 0.0),
                                 market_state)
        horizon[c] = FIRST_YIELD[c] if ONGOING[c] else MAX_YIELD_DAY[c]
    ranked = [c for c in CROPS_LIST if profits.get(c, 0) > 0]
    ranked.sort(key=lambda c: profits[c] / max(1, horizon[c]), reverse=True)

    plan = {}
    remaining = tiles_avail

    # Early-game cash block: cheap fast crops so money arrives in time for
    # land and animals.  Wheat/carrot pay out from day 2-4 while premium
    # crops sit dormant until day 8-13.  Only relevant for the first days:
    # by day 5 melons are in their water window and no longer need cheap
    # cash-flow crops competing for the same tiles.
    cheap = [c for c in ("WHEAT", "CARROT") if profits.get(c, 0) > 0]
    if day <= 4 and money < 2500 and cheap:
        n_cheap = max(4, int(tiles_avail * 0.35))
        for c in cheap:
            if n_cheap <= 0:
                break
            take = min(n_cheap, remaining)
            plan[c] = take
            remaining -= take
            n_cheap -= take

    for c in ranked:
        if remaining <= 0:
            break
        cap = min(allowance_tiles(c, day, prices, inv, params,
                                  market_state["town"], opp, my_rate.get(c, 0.0)),
                  HARD_CAPS[c])
        take = int(min(cap, remaining))
        if take <= 0:
            continue
        plan[c] = take
        remaining -= take

    fert_reserve = 0
    for c, n in plan.items():
        if c == "STRAWBERRY" and _fert_marginal(c, day, prices, params, market_state) > 0:
            fert_reserve += 3 * n
        elif c == "TOMATO" and _fert_marginal(c, day, prices, params, market_state) > 0:
            fert_reserve += 1 * n
    for (x, y, t) in farm_state["plants"]:
        c = t["crop"]
        if c == "STRAWBERRY" and _fert_marginal(c, day, prices, params, market_state) > 0 \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 2
        elif c == "TOMATO" and _fert_marginal(c, day, prices, params, market_state) > 0 \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 1

    return {"crops": plan, "ranked": ranked, "profit": profits,
            "fert_reserve": min(fert_reserve, FERT_USE_MAX * 2),
            "tiles_avail": tiles_avail, "labor_spare": labor_spare,
            "labor_capacity": capacity}


# ============================================================================
# Animal economics
# ============================================================================
def animal_value(animal, day, prices, wheat_cost, fert_est, farm_state,
                 market_state, opp, inv, params):
    """Expected net contribution of buying `animal` today.

    revenue      = days alive x (base + expected CARE bonus) x projected price
    costs        = feed wheat + labor (feed/care/harvest/build/placement)
                 + animal capital
    plus         = fertilizer collection value
    The market projection includes the opponent's and our own committed
    supply, so milk/wool/egg gluts damp animal profitability too.
    """
    placed_day = day + 1
    n_prod = _animal_prods_left(animal, placed_day, day)
    if n_prod <= 1:
        return -1e9
    p = ANIMAL_PRODUCT[animal]
    base_rate = animal_base_rate(animal)          # units/day (no CARE)
    care_rate = base_rate * care_probability(farm_state, day)
    days_alive = max(1, int(n_prod * ANIMAL_INTERVAL[animal]))
    horizon = min(LAST_HARVEST - day, days_alive)
    town = market_state["town"]
    my_rate = my_supply_rates(farm_state, day).get(p, 0.0) \
        + base_rate * (1.0 + care_probability(farm_state, day))
    opp_rate = opp["daily_rate"].get(p, 0.0)
    est_p = project_price(p, inv.get(p, params[p]["I0"]), day, horizon,
                          town, opp_rate, params, my_rate)
    # Revenue is per-day rate x days alive: a cow produces 2 units per
    # 2-day tick = 1/day base + expected care bonus, paid out over its
    # whole remaining life.
    revenue = days_alive * (base_rate + care_rate) * est_p
    feed_cost = days_alive * wheat_cost
    labor_cost = (days_alive * 2.0 + n_prod + 10.0) * LABOR_VALUE
    fert_value = fert_est * days_alive * 0.5
    return (revenue + fert_value - feed_cost - labor_cost) * 0.85 \
        - ANIMAL_COST[animal]


def animal_plan(farm_state, market_state, opp, day, prices, inv, params, money, slots):
    orders = []
    if not ANIMALS_ENABLED or day > 14:
        return orders
    fert_price = prices.get("FERTILIZER", 100)
    if fert_price < FERT_PRICE_MIN:
        return orders
    days_left = LAST_DAY - day

    wheat_price = prices.get("WHEAT", 25)
    opp_fert = opp["daily_rate"].get("FERTILIZER", 0.0) * days_left
    my_fert_daily = farm_state["n_animals"] * 0.8
    fert_inv_f = max(1.0, inv.get("FERTILIZER", 10000)
                     + my_fert_daily * days_left * 0.7 + opp_fert)
    fert_final = price_at("FERTILIZER", fert_inv_f, params)
    fert_est = (fert_price + fert_final) / 2.0
    if fert_est < FERT_PRICE_MIN:
        return orders

    feed_cap = farm_state["crop_counts"]["WHEAT"] * 1.2 \
        + market_state["shed"].get("WHEAT", 0) / 8.0 \
        + max(0, money - 400) / 400.0
    feed_cap = max(2.0, min(feed_cap, 16.0))

    total_animals = farm_state["n_animals"] + sum(
        market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
    pending = sum(market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
    placement_room = farm_state["empty_coop"] + farm_state["empty_pasture"] \
        + farm_state["empty"]

    for a in ("GOOSE", "COW", "SHEEP"):
        have = farm_state["animal_counts"][a] + market_state["shed"].get(a, 0)
        cap = min(ANIMAL_CAPS[a], int(feed_cap))
        if have >= cap or slots <= 0:
            continue
        if pending >= 6:
            break
        min_day = 4 if a == "GOOSE" else (8 if a == "COW" else 9)
        if day < min_day:
            continue
        val = animal_value(a, day, prices, wheat_price * 0.9, fert_est,
                           farm_state, market_state, opp, inv, params)
        if val < ANIMAL_COST[a] * 0.5:
            continue
        if total_animals >= placement_room:
            continue
        struct = ANIMAL_STRUCT[a]
        cost = ANIMAL_COST[a]
        if money - cost < CASH_FLOOR + 300:
            continue
        orders.append(["BUY_ANIMAL", a, 1])
        slots -= 1
        money -= cost
        total_animals += 1
    return orders


# ============================================================================
# Selling
# ============================================================================
def _safe_batch(item, q, inv, params, price):
    """Largest batch to sell this turn without dragging the average price
    below 0.8x the current price."""
    n = 1
    while n < q and batch_avg(item, n, inv, params) >= 0.8 * price:
        n += 1
    return max(1, n - 1)


def sell_plan(farm_state, market_state, opp, day, prices, inv, params):
    orders = []
    shed = market_state["shed"]
    slots = MAX_ORDERS
    shed_total = sum(shed.values())
    money = market_state["money"]
    endgame = day >= ENDGAME_DAY
    forced = shed_total >= SHED_FORCE or endgame
    cash_need = money < 150
    reserve_wheat = farm_state["n_animals"] + 5 if farm_state["n_animals"] else 0
    fert_reserve = market_state["fert_reserve"]
    days_left = max(1, LAST_DAY - day)

    for p in SELL_ORDER:
        if slots <= 0:
            break
        if not forced and slots <= 3:
            break                    # keep slots for buys
        q = shed.get(p, 0)
        if p == "WHEAT":
            q = max(0, q - reserve_wheat)
        elif p == "FERTILIZER":
            q = max(0, q - fert_reserve)
            if not (forced or endgame) and day < ENDGAME_DAY - 1:
                continue
        if q <= 0:
            continue
        price = prices.get(p, BASE_PRICE[p])
        inv_p = inv.get(p, params[p]["I0"])
        trend = price_trend(p, day)

        if forced or cash_need:
            n = q
        else:
            drain = town_rate(day, p, market_state["town"])
            opp_sell = opp["daily_rate"].get(p, 0.0) * OPP_SELL_FRACTION
            my_sell = my_supply_rates(farm_state, day).get(p, 0.0)
            per_turn = max(0.0, drain - opp_sell - my_sell) / 24.0
            pace = q / max(1.0, days_left * 24.0)      # shed empties by endgame
            if price >= BASE_PRICE[p]:
                n = min(q, pace + per_turn + 0.5)
            elif price >= 0.8 * BASE_PRICE[p]:
                n = min(q, (pace + per_turn) * 0.7)
            else:
                # crashed to the floor: price will not recover, dump it
                n = q
            # Price-trend adjustment: recoveries deserve a hold (unless the
            # product's curve is a premium one that collapses on gluts),
            # deteriorating markets deserve faster exit.
            if trend is not None and price < BASE_PRICE[p]:
                if trend > 2.0 and p not in PREMIUM:
                    n *= 0.5
                elif trend < -2.0:
                    n = min(q, max(n, (pace + per_turn) * 1.5))
            n = min(n, _safe_batch(p, q, inv_p, params, price))
            n = max(1.0, n)
        if n >= 1:
            orders.append(["SELL", p, int(n)])
            slots -= 1
    return orders


# ============================================================================
# Investments: seeds / fertilizer / wheat / hands / land
# ============================================================================
def seed_plan(crop_plan_result, farm_state, money, slots, day):
    orders = []
    plan = crop_plan_result["crops"]
    ranked = crop_plan_result["ranked"]
    planted = farm_state["crop_counts"]
    if money < CASH_FLOOR + 30 or slots <= 0:
        return orders
    land_reserve = 0
    n_extra = max(0, farm_state["size"] // 5 - 1)
    if n_extra < 3 and LAND_DAYS[n_extra] - 1 <= day <= LAND_LAST_DAY:
        cost = LAND_CASH[n_extra]
        if money - cost >= 250:
            land_reserve = cost
    budget = max(0, money - land_reserve - CASH_FLOOR)
    for c in ranked:
        if slots <= 0 or budget <= 0:
            break
        target = plan.get(c, 0)
        have = farm_state["seeds"].get(c, 0) + planted.get(c, 0)
        deficit = target - have
        if deficit <= 0:
            continue
        cost = SEED_COST[c]
        if cost > budget:
            continue
        ramp = 12 if cost <= 20 else (8 if cost <= 60 else 5)
        want = min(deficit, ramp)
        buy = int(min(want, budget // cost))
        if buy <= 0 or slots <= 0:
            continue
        # One aggregated order per crop: the env processes the quantity
        # unit-by-unit internally, so BUY_SEED WHEAT 10 costs one of the
        # max-10 order slots instead of ten.
        orders.append(["BUY_SEED", c, buy])
        slots -= 1
        budget -= cost * buy
    return orders


def fert_buy_plan(farm_state, market_state, prices, params, money, slots):
    orders = []
    shed_fert = market_state["shed"].get("FERTILIZER", 0)
    need = market_state["fert_reserve"]
    if shed_fert >= need or slots <= 0 or money < 400:
        return orders
    best = 0.0
    for c in ("STRAWBERRY", "TOMATO"):
        m = _fert_marginal(c, 0, prices, params, market_state)
        best = max(best, m)
    fert_price = prices.get("FERTILIZER", 100)
    if best > fert_price * 1.3:
        n = min(2, need - shed_fert)
        for _ in range(n):
            if slots <= 0:
                break
            orders.append(["BUY_PRODUCT", "FERTILIZER", 1])
            slots -= 1
    return orders


def wheat_buy_plan(farm_state, market_state, prices, money, slots):
    orders = []
    if farm_state["n_animals"] <= 0 or slots <= 0:
        return orders
    shed_wheat = market_state["shed"].get("WHEAT", 0)
    need = farm_state["n_animals"] + 8
    deficit = need - shed_wheat
    if deficit > 0 and money > 200 and sum(market_state["shed"].values()) < 88:
        for _ in range(min(6, deficit)):
            if slots <= 0:
                break
            orders.append(["BUY_PRODUCT", "WHEAT", 1])
            slots -= 1
    return orders


def hire_plan(farm_state, market_state, day, hour, money, slots,
              plan_value_per_action, labor_spare=None,
              labor_capacity=None):
    orders = []
    if day >= 29 or hour >= HIRE_HOUR_CUT or slots <= 0 or money < 60:
        return orders
    hires_today = farm_state["hires_today"]
    overload = 0.0
    if labor_spare is not None and labor_capacity is not None:
        overload = max(0.0, -labor_spare)

    # Economic target: hire hands while the marginal hand's output (about
    # ACTIONS_PER_HAND effective actions/day of the marginal task value) beats
    # its Fibonacci wage.  No estimate -> fall back to a cash-tiered ramp.
    target = 0
    if plan_value_per_action is not None and plan_value_per_action > 0:
        while target < HANDS_MAX:
            cost = FIB[min(target, len(FIB) - 1)]
            if ACTIONS_PER_HAND * plan_value_per_action * 0.6 < cost:
                break
            target += 1
    # Workload target: hands are also hired because concrete tasks are
    # waiting (same tile-demand model as the scheduler), not only because
    # planting is profitable.
    if target < 2 or plan_value_per_action is None:
        if money < 200:
            target = max(target, 2)
        elif money < 600:
            target = max(target, 3)
        elif money < 1200:
            target = max(target, 4)
        elif money < 2500:
            target = max(target, 5)
        else:
            target = max(target, 6)
    # Overloaded farm: tasks are waiting in a queue, extra hands are pure win
    if overload >= ACTIONS_PER_HAND and plan_value_per_action is not None \
            and plan_value_per_action > HAND_VALUE_MIN:
        target = HANDS_MAX

    while hires_today < target and slots > 0:
        cost = FIB[min(hires_today, len(FIB) - 1)]
        if money - cost < CASH_FLOOR:
            break
        orders.append(["HIRE"])
        slots -= 1
        hires_today += 1
        money -= cost
    return orders


def _land_mix_value(plan, day, prices, inv, params, town, opp, my_rate):
    """Expected per-tile value of the crop mix a new 25-tile quadrant would
    host: allocate tiles to the plan's ranked crops exactly like crop_plan
    does (allowance- and HARD_CAPS-bounded), then average the profits.  New
    land is not worth min(profits): its tiles take the best allowed crops."""
    ranked = plan["ranked"]
    if not ranked:
        return 0.0
    remaining = 25
    total = 0.0
    for c in ranked:
        if remaining <= 0:
            break
        cap = min(allowance_tiles(c, day, prices, inv, params, town, opp,
                                  my_rate.get(c, 0.0)), HARD_CAPS[c])
        take = min(max(0, int(cap) - plan["crops"].get(c, 0)), remaining)
        total += take * plan["profit"].get(c, 0.0)
        remaining -= take
    if total <= 0:
        return 0.0
    return total / 25.0


def land_plan(farm_state, market_state, day, money, slots, plan, prices, inv,
              params, opp, labor_spare=None, labor_capacity=None):
    orders = []
    n_extra = max(0, len(market_state["unlocked"]) - 1)
    if n_extra >= 3 or day > LAND_LAST_DAY or slots <= 0:
        return orders
    cost = LAND_CASH[n_extra]
    if money < cost or day < LAND_DAYS[n_extra]:
        return orders
    my_rate = my_supply_rates(farm_state, day)
    mix = _land_mix_value(plan, day, prices, inv, params, market_state["town"],
                          opp, my_rate)
    if mix <= 0:
        return orders
    # Animals beat land: once we own animals waiting to be placed, spend cash
    # on them instead of a third (or later) quadrant.
    pending = sum(market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
    if day >= 12 and (pending > 0 or farm_state["n_animals"] > 0):
        return orders
    # Forward projection of the marginal quadrant: clone the farm with the
    # mix planted today and diff the projected bank, then scale by how much
    # season remains (late land has fewer harvest waves) and by available
    # labor (an overloaded farm cannot tend new land).
    remaining = max(1, LAST_DAY - day + 1)
    horizon_factor = min(1.0, remaining / 14.0)
    fs2 = dict(farm_state)
    fs2["plants"] = list(farm_state["plants"])
    fs2["crop_counts"] = dict(farm_state["crop_counts"])
    remaining25 = 25
    for c in plan["ranked"]:
        if remaining25 <= 0:
            break
        cap = min(allowance_tiles(c, day, prices, inv, params,
                                  market_state["town"], opp,
                                  my_rate.get(c, 0.0)), HARD_CAPS[c])
        take = min(max(0, int(cap) - plan["crops"].get(c, 0)), remaining25)
        for _ in range(take):
            fs2["plants"].append((0, 0, {"crop": c, "planted_day": day}))
            fs2["crop_counts"][c] += 1
        remaining25 -= take
    base = forward_bank(day, 9, prices, inv, market_state["town"], opp,
                        params, farm_state, market_state["shed"])
    with_land = forward_bank(day, 9, prices, inv, market_state["town"], opp,
                             params, fs2, market_state["shed"])
    gain = (with_land - base) * horizon_factor
    if labor_capacity is not None and labor_spare is not None:
        new_land_actions = 25 * 2.5
        if labor_spare < new_land_actions:
            gain *= max(0.3, labor_spare / new_land_actions)
    used = sum(farm_state["crop_counts"].values()) + farm_state["n_animals"] \
        + farm_state["empty_coop"] + farm_state["empty_pasture"]
    usable = 25 * len(market_state["unlocked"]) - 4    # shed overlaps board
    if gain >= cost and used >= 0.6 * usable:
        orders.append(["BUY_LAND"])
    return orders


# ============================================================================
# Task scheduler
# ============================================================================
def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _value_bonus(value, scale=8, cap=700):
    return max(0, min(cap, int(value) // scale))


def _fert_needed(market_state):
    reserve = market_state.get("fert_reserve", 0)
    if reserve <= 0:
        return 0
    have = market_state["shed"].get("FERTILIZER", 0)
    for inv in market_state["inventories"]:
        have += inv.get("FERTILIZER", 0)
    return max(0, reserve - have)


def _in_window(tile, day):
    crop = tile["crop"]
    if ONGOING[crop]:
        return False
    start = (MAX_YIELD_DAY[crop] + 1) // 2
    age = day - tile.get("planted_day", day)
    return start <= age <= MAX_YIELD_DAY[crop]


def _wheat_water_bonus(tile, day, fert_worth):
    bonus = 2 if tile.get("fertilized_until_day", -1) >= day \
        and fert_worth.get(tile["crop"], 0) > 0 else 1
    return bonus


def build_tasks(farm, farm_state, plan, market_state, day, prices):
    """Returns (tasks, plant_map, build_map). tasks: (x, y, action, arg,
    prio); plant_map: {tile: crop}; build_map: {tile: BUILD_*}."""
    tiles = farm["tiles"]
    size = farm_state["size"]
    tasks = []
    plant_map = {}
    build_map = {}

    plan_crops = sorted(plan["crops"].keys(),
                        key=lambda c: plan["profit"].get(c, 0), reverse=True)
    remaining = dict(plan["crops"])
    seed_budget = {c: farm_state["seeds"].get(c, 0) for c in CROPS_LIST}

    empty_cells = [(x, y) for y in range(size) for x in range(size)
                   if tiles[y][x] is None]
    empty_cells.sort(key=lambda p: _manhattan(size // 2, size // 2, p[0], p[1]))
    for (x, y) in empty_cells:
        for c in plan_crops:
            if remaining.get(c, 0) > 0 and seed_budget.get(c, 0) > 0:
                plant_map[(x, y)] = c
                remaining[c] -= 1
                seed_budget[c] -= 1
                break

    fert_worth = {c: _fert_marginal(c, day, prices, market_state["params"],
                                    market_state)
                  for c in CROPS_LIST}

    shed_tiles = market_state["shed_tiles"]
    need_feed = farm_state["unfed"] > 0
    any_fert_need = any(fert_worth[c] > 0 and plan["crops"].get(c, 0) > 0
                        for c in plan_crops)
    if not any_fert_need:
        for (x, y, t) in farm_state["plants"]:
            c = t["crop"]
            if fert_worth.get(c, 0) > 0 and t.get("fertilized_until_day", -1) < day:
                any_fert_need = True
                break

    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if t is None:
                crop = plant_map.get((x, y))
                if crop:
                    prio = P_PLANT + _value_bonus(plan["profit"].get(crop, 0) // 2)
                    tasks.append((x, y, "PLANT", crop, prio))
            elif not isinstance(t, dict):
                continue
            elif t.get("kind") == "WEED":
                prio = P_DIG_WEED + (200 if (x, y) in plant_map else 0)
                tasks.append((x, y, "DIG", None, prio))
            elif t.get("kind") == "PLANT":
                crop = t["crop"]
                ripe = t.get("yield_units", 0) > 0
                if ONGOING[crop]:
                    if ripe and t["yield_units"] >= 2:
                        val = t["yield_units"] * prices.get(crop, BASE_PRICE[crop])
                        tasks.append((x, y, "HARVEST", crop,
                                      P_HARVEST + _value_bonus(val)))
                    elif not t.get("watered_today", True):
                        if t.get("consecutive_unwatered", 0) >= 1:
                            prio = P_WATER_CRIT
                        elif t.get("fertilized_until_day", -1) >= day:
                            protect = _productions_remaining(crop, t["planted_day"], day) \
                                * prices.get(crop, BASE_PRICE[crop])
                            prio = P_WATER + _value_bonus(protect // 4)
                        else:
                            protect = _productions_remaining(crop, t["planted_day"], day) \
                                * prices.get(crop, BASE_PRICE[crop])
                            prio = P_WATER + _value_bonus(protect // 8)
                        tasks.append((x, y, "WATER", crop, prio))
                    elif _productions_remaining(crop, t["planted_day"], day) == 0 \
                            and t.get("yield_units", 0) <= 0:
                        tasks.append((x, y, "DIG", crop, P_DIG_SPENT))
                    elif fert_worth.get(crop, 0) > 0 \
                            and t.get("fertilized_until_day", -1) < day:
                        tasks.append((x, y, "FERTILIZE", crop,
                                      P_FERTILIZE + _value_bonus(fert_worth[crop])))
                else:
                    age = day - t.get("planted_day", day)
                    if ripe and age >= MAX_YIELD_DAY[crop]:
                        val = t["yield_units"] * prices.get(crop, BASE_PRICE[crop]) \
                            + max(0, plan["profit"].get(crop, 0))
                        tasks.append((x, y, "HARVEST", crop,
                                      P_HARVEST + _value_bonus(val)))
                    elif not t.get("watered_today", True):
                        if t.get("consecutive_unwatered", 0) >= 1:
                            prio = P_WATER_CRIT
                        elif _in_window(t, day):
                            bonus = _wheat_water_bonus(t, day, fert_worth)
                            val = bonus * prices.get(crop, BASE_PRICE[crop])
                            prio = P_WATER_WINDOW + _value_bonus(val)
                        else:
                            val = t.get("yield_units", 1) \
                                * prices.get(crop, BASE_PRICE[crop])
                            prio = P_WATER + _value_bonus(val // 8)
                        tasks.append((x, y, "WATER", crop, prio))
            elif "animal" in t:
                crop = t["animal"]
                if t.get("consecutive_unfed", 0) >= 1:
                    tasks.append((x, y, "FEED", crop, P_FEED_CRIT))
                elif t.get("yield_units", 0) >= 2 \
                        or t.get("yield_units", 0) >= ANIMAL_HELD[crop] - 1:
                    p = ANIMAL_PRODUCT[crop]
                    val = t["yield_units"] * prices.get(p, BASE_PRICE[p])
                    tasks.append((x, y, "HARVEST", crop,
                                  P_HARVEST + _value_bonus(val)))
                elif not t.get("fed_today", False):
                    tasks.append((x, y, "FEED", crop, P_FEED))
                elif t.get("fertilizer_available", False) \
                        and _fert_needed(market_state) > 0:
                    tasks.append((x, y, "COLLECT", crop,
                                  P_COLLECT + _value_bonus(
                                      prices.get("FERTILIZER", 100))))
                elif not t.get("cared_today", False) and t.get("fed_today", False):
                    p = ANIMAL_PRODUCT[crop]
                    if _animal_prods_left(crop, t.get("placed_day", day), day) > 0 \
                            and prices.get(p, BASE_PRICE[p]) \
                            / ANIMAL_INTERVAL[crop] > LABOR_VALUE:
                        tasks.append((x, y, "CARE", crop,
                                      P_CARE + _value_bonus(
                                          prices.get(p, BASE_PRICE[p]) * 0.7)))
            elif t.get("kind") in ("COOP", "PASTURE") and "animal" not in t:
                for a in ANIMALS_LIST:
                    if ANIMAL_STRUCT[a] == t["kind"]:
                        in_inv = any(inv.get(a, 0) > 0
                                     for inv in market_state["inventories"])
                        if market_state["shed"].get(a, 0) > 0 or in_inv:
                            tasks.append((x, y, "PLACE", a, P_PLACE))
                        break

    # structure building for pending animal purchases: one structure per
    # pending animal, each with a scheduled BUILD task
    build_cells = [p for p in empty_cells if p not in plant_map]
    for a in ANIMALS_LIST:
        struct = ANIMAL_STRUCT[a]
        if struct == "COOP":
            empty_struct = farm_state["empty_coop"]
        else:
            empty_struct = farm_state["empty_pasture"]
        have = market_state["shed"].get(a, 0) + sum(
            inv.get(a, 0) for inv in market_state["inventories"])
        pending = max(0, have - empty_struct)
        for _ in range(pending):
            if not build_cells:
                break
            (bx, by) = build_cells.pop(0)
            build_map[(bx, by)] = "BUILD_COOP" if struct == "COOP" else "BUILD_PASTURE"
            tasks.append((bx, by, "BUILD_COOP" if struct == "COOP" else "BUILD_PASTURE",
                          None, P_BUILD))

    n_build_coop = sum(1 for t in tasks if t[2] == "BUILD_COOP")
    n_build_pasture = sum(1 for t in tasks if t[2] == "BUILD_PASTURE")
    struct_room = {
        "GOOSE": farm_state["empty_coop"] + n_build_coop,
        "COW": farm_state["empty_pasture"] + n_build_pasture,
        "SHEEP": farm_state["empty_pasture"] + n_build_pasture,
    }

    # pickups: fertilizer / wheat / animals
    n_units = 1 + len(farm.get("hands", []) or [])
    shed_tiles_list = sorted(shed_tiles,
                             key=lambda p: (p[1], p[0]))
    any_fert_carried = any(inv.get("FERTILIZER", 0) > 0
                           for inv in market_state["inventories"])
    n_fert_tasks = sum(1 for (x, y, act, arg, prio) in tasks if act == "FERTILIZE")
    if any_fert_need and not any_fert_carried \
            and market_state["shed"].get("FERTILIZER", 0) > 0:
        for u in range(min(n_units, max(1, (n_fert_tasks + 2) // 3))):
            sx, sy = market_state["farmer"]
            st = shed_tiles_list[u % len(shed_tiles_list)]
            tasks.append((st[0], st[1], "PICKUP_FERT", 3, P_PICKUP_FERT))

    any_wheat_carried = any(inv.get("WHEAT", 0) > 0
                            for inv in market_state["inventories"])
    need = farm_state["n_animals"]
    shed_wheat = market_state["shed"].get("WHEAT", 0)
    need_feed_wheat = farm_state["unfed"] - sum(
        inv.get("WHEAT", 0) for inv in market_state["inventories"])
    if need_feed_wheat > 0 and shed_wheat > 0:
        # parallel feed: hand each unit its own small batch so animals are
        # fed before they escape
        n_tasks = min(n_units, max(1, need_feed_wheat))
        per = (need_feed_wheat + n_tasks - 1) // n_tasks
        for u in range(n_tasks):
            sx, sy = market_state["farmer"]
            st = shed_tiles_list[u % len(shed_tiles_list)]
            n = min(per, shed_wheat)
            if n <= 0:
                break
            tasks.append((st[0], st[1], "PICKUP_WHEAT", n, P_PICKUP_WHEAT))
            shed_wheat -= n

    for a in ANIMALS_LIST:
        in_inv = any(inv.get(a, 0) > 0 for inv in market_state["inventories"])
        if market_state["shed"].get(a, 0) > 0 and not in_inv \
                and struct_room.get(a, 0) > 0:
            sx, sy = market_state["farmer"]
            st = shed_tiles_list[0]
            tasks.append((st[0], st[1], "PICKUP_ANIMAL", a, P_PICKUP_ANIMAL))

    return tasks, plant_map, build_map


def choose_target(tasks, ux, uy, claimed, unit_bias, my_inv=None):
    best, best_score = None, None
    carries_animal = bool(my_inv) and any(my_inv.get(a, 0) > 0 for a in ANIMALS_LIST)
    carries_wheat = bool(my_inv) and my_inv.get("WHEAT", 0) > 0
    has_feed_tasks = any(act == "FEED" for (x, y, act, arg, prio) in tasks)
    for (x, y, act, arg, prio) in tasks:
        if (x, y) in claimed:
            continue
        if carries_animal and act != "PLACE":
            continue
        if carries_wheat and has_feed_tasks and act != "FEED":
            continue
        if act == "FEED" and (not my_inv or my_inv.get("WHEAT", 0) <= 0):
            continue
        if act == "PLACE" and (not my_inv or my_inv.get(arg, 0) <= 0):
            continue
        # task_score = urgency x value with travel cost scaled by the task
        # band: rescue/feed/harvest tasks travel freely (their value dwarfs
        # a few movement turns), low-value chores stay near the unit.
        dist = _manhattan(ux, uy, x, y)
        if prio >= P_WATER_CRIT:
            travel_pen = TRAVEL_COST * dist * 0.4
        elif prio >= P_HARVEST:
            travel_pen = TRAVEL_COST * dist * 0.7
        else:
            travel_pen = TRAVEL_COST * dist
        score = prio - travel_pen \
            - REGION_BIAS * _manhattan(unit_bias[0], unit_bias[1], x, y)
        if best_score is None or score > best_score:
            best_score, best = score, (x, y, act, arg, prio)
    if best:
        claimed.add((best[0], best[1]))
    return best


def unit_action(ux, uy, tile, my_inv, farm_state, market_state, plan,
                budget, claimed, unit_bias, tasks, plant_map, build_map,
                day, prices):
    shed_tiles = market_state["shed_tiles"]
    params = market_state["params"]
    need_feed = farm_state["unfed"] > 0
    carrying = any(my_inv.get(a, 0) > 0 for a in ANIMALS_LIST) \
        or (need_feed and my_inv.get("WHEAT", 0) > 0)

    if tile is not None and isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT" and not carrying:
            crop = tile["crop"]
            ripe = tile.get("yield_units", 0) > 0
            age = day - tile.get("planted_day", day)
            if ONGOING[crop]:
                if ripe and tile["yield_units"] >= 2:
                    return ["HARVEST"]
                if not tile.get("watered_today", True):
                    return ["WATER"]
                if _productions_remaining(crop, tile["planted_day"], day) == 0 \
                        and tile.get("yield_units", 0) <= 0:
                    return ["DIG"]
                if my_inv.get("FERTILIZER", 0) > 0 \
                        and tile.get("fertilized_until_day", -1) < day \
                        and _fert_marginal(crop, day, prices, params, market_state) > 0:
                    return ["FERTILIZE"]
            else:
                if ripe and age >= MAX_YIELD_DAY[crop]:
                    return ["HARVEST"]
                if not tile.get("watered_today", True):
                    return ["WATER"]
                if my_inv.get("FERTILIZER", 0) > 0 \
                        and tile.get("fertilized_until_day", -1) < day \
                        and _fert_marginal(crop, day, prices, params, market_state) > 0:
                    return ["FERTILIZE"]
        elif kind == "WEED" and not carrying:
            return ["DIG"]
        elif "animal" in tile:
            crop = tile["animal"]
            if tile.get("consecutive_unfed", 0) >= 1 \
                    and not tile.get("fed_today", False) \
                    and my_inv.get("WHEAT", 0) > 0:
                return ["FEED"]
            if tile.get("yield_units", 0) >= 2 \
                    or tile.get("yield_units", 0) >= ANIMAL_HELD[crop] - 1:
                return ["HARVEST"]
            if not tile.get("fed_today", False) and my_inv.get("WHEAT", 0) > 0:
                return ["FEED"]
            if tile.get("fertilizer_available", False) \
                    and _fert_needed(market_state) > 0:
                return ["COLLECT"]
            if not tile.get("cared_today", False) and tile.get("fed_today", False) \
                    and _animal_prods_left(crop, tile.get("placed_day", day), day) > 0 \
                    and prices.get(ANIMAL_PRODUCT[crop],
                                   BASE_PRICE[ANIMAL_PRODUCT[crop]]) \
                    / ANIMAL_INTERVAL[crop] > LABOR_VALUE:
                return ["CARE"]
        elif kind in ("COOP", "PASTURE") and "animal" not in tile:
            for a in ANIMALS_LIST:
                if ANIMAL_STRUCT[a] == kind and my_inv.get(a, 0) > 0:
                    return ["PLACE", a]
    elif tile is None and not carrying:
        build = build_map.get((ux, uy))
        if build:
            return [build]
        crop = plant_map.get((ux, uy))
        if crop and budget.get(crop, 0) > 0:
            budget[crop] -= 1
            return ["PLANT", crop]

    if (ux, uy) in shed_tiles and (ux, uy) not in claimed and not carrying:
        for (x, y, act, arg, prio) in tasks:
            if (x, y) == (ux, uy):
                if act == "PICKUP_FERT":
                    return ["PICKUP", "FERTILIZER", int(arg)]
                if act == "PICKUP_WHEAT":
                    n = min(int(arg), market_state["shed"].get("WHEAT", 0))
                    return ["PICKUP", "WHEAT", n]
                if act == "PICKUP_ANIMAL":
                    return ["PICKUP", arg, 1]

    tgt = choose_target(tasks, ux, uy, claimed, unit_bias, my_inv)
    if tgt:
        step = _step_toward(ux, uy, tgt[0], tgt[1])
        if step:
            return [step]
    return ["PASS"]


def schedule_units(farm, farm_state, market_state, plan, tasks, plant_map,
                   build_map, day, prices):
    size = farm_state["size"]
    half = size // 2
    h2 = half // 2
    anchors = [(h2, h2), (3 * h2, h2), (h2, 3 * h2), (3 * h2, 3 * h2)]

    farmer_x, farmer_y = farm["farmer"]
    budget = {c: farm_state["seeds"].get(c, 0) for c in CROPS_LIST}
    claimed = set()
    farmer_inv = market_state["inventories"][0] if market_state["inventories"] else {}

    fa = unit_action(farmer_x, farmer_y, farm["tiles"][farmer_y][farmer_x],
                     farmer_inv, farm_state, market_state, plan, budget,
                     claimed, anchors[0], tasks, plant_map, build_map,
                     day, prices)
    if fa and fa[0] not in ("NORTH", "SOUTH", "EAST", "WEST", "PASS"):
        claimed.add((farmer_x, farmer_y))

    hand_actions = []
    for h_idx, (hx, hy) in enumerate(farm.get("hands", []) or []):
        h_inv = market_state["inventories"][h_idx + 1] \
            if h_idx + 1 < len(market_state["inventories"]) else {}
        bias = anchors[(h_idx + 1) % 4]
        ha = unit_action(hx, hy, farm["tiles"][hy][hx], h_inv, farm_state,
                         market_state, plan, budget, claimed, bias, tasks,
                         plant_map, build_map, day, prices)
        if ha and ha[0] not in ("NORTH", "SOUTH", "EAST", "WEST", "PASS"):
            claimed.add((hx, hy))
        hand_actions.append(ha)

    return fa, hand_actions


# ============================================================================
# Agent entry point
# ============================================================================
def agent(obs, config=None):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    tiles = farm["tiles"]
    size = len(tiles)
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    money = farm.get("money", 0)
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    inventories = private.get("inventories", [{}]) or [{}]
    market_obs = obs.get("market", {}) or {}
    prices = market_obs.get("prices", {}) or {}
    inv = market_obs.get("inventory", {}) or {}
    town = obs.get("town", {}) or {}
    unlocked = set(farm.get("unlocked_quadrants", []) or [])
    hires_today = farm.get("hires_today", 0)
    params = _params(market_obs)

    half = size // 2
    shed_tiles = {(half - 1, half - 1), (half, half - 1),
                  (half - 1, half), (half, half)}

    farm_state = analyze_farm(farm, day)
    farm_state["seeds"] = seeds
    farm_state["hires_today"] = hires_today
    farm_state["n_units"] = 1 + len(farm.get("hands", []) or [])

    opp = analyze_opponent(farms, player, day)

    market_state = {
        "shed": shed,
        "inventories": inventories,
        "town": town,
        "money": money,
        "size": size,
        "unlocked": unlocked,
        "shed_tiles": shed_tiles,
        "farmer": farm["farmer"],
        "params": params,
        "fert_reserve": 0,
    }

    if hour == 0:
        if day == 0:
            _PRICE_HIST.clear()
        for p in PRODUCTS:
            if p in prices:
                hist = _PRICE_HIST.setdefault(p, [])
                if not hist or hist[-1][0] < day:
                    hist.append((day, prices[p]))
                hist[:] = hist[-6:]

    plan = crop_plan(farm_state, market_state, opp, day, prices, inv,
                     params, money)
    market_state["fert_reserve"] = plan["fert_reserve"]

    # ---- market orders ---------------------------------------------------
    orders = []
    slots = MAX_ORDERS

    sell_orders = sell_plan(farm_state, market_state, opp, day, prices, inv,
                            params)
    sell_cash = 0.0
    for o in sell_orders:
        if o[0] == "SELL":
            sell_cash += o[2] * batch_avg(o[1], o[2], inv.get(o[1], 10000), params)
    orders.extend(sell_orders)
    slots -= len(sell_orders)

    money_proj = money + sell_cash

    total_plan_actions = max(1, sum(plan["crops"].values()) * 6
                             + ANIMALS_ENABLED * farm_state["n_animals"] * 2)
    total_plan_value = sum(plan["crops"].get(c, 0) * plan["profit"].get(c, 0)
                           for c in plan["crops"])
    plan_value_per_action = total_plan_value / total_plan_actions \
        if total_plan_value > 0 else None

    # ---- hire first: hands are the cheapest labor in the game --------------
    hire_orders = hire_plan(farm_state, market_state, day, hour, money_proj,
                            slots, plan_value_per_action,
                            plan.get("labor_spare"), plan.get("labor_capacity"))
    orders.extend(hire_orders)
    slots -= len(hire_orders)

    # ---- land: buying land expands the crop plan for today -----------------
    land_orders = land_plan(farm_state, market_state, day, money_proj, slots,
                            plan, prices, inv, params, opp,
                            plan.get("labor_spare"), plan.get("labor_capacity"))
    if land_orders:
        n_extra = max(0, len(market_state["unlocked"]) - 1)
        money_proj -= LAND_CASH[n_extra]
        plan = crop_plan(farm_state, market_state, opp, day, prices, inv,
                         params, money_proj, land_extra=24)
        market_state["fert_reserve"] = plan["fert_reserve"]
    orders.extend(land_orders)
    slots -= len(land_orders)

    # ---- animals rank ahead of seeds: their ROI per order slot is higher
    # and they are capped, so they must not starve when slots are tight -----
    if ANIMALS_ENABLED:
        animal_b = animal_plan(farm_state, market_state, opp, day, prices,
                               inv, params, money_proj, slots)
        orders.extend(animal_b)
        slots -= len(animal_b)
        money_proj -= sum(ANIMAL_COST[o[1]] for o in animal_b)

    seed_orders = seed_plan(plan, farm_state, money_proj, slots, day)
    orders.extend(seed_orders)
    slots -= len(seed_orders)
    money_proj -= sum(SEED_COST[o[1]] * o[2] for o in seed_orders)

    if ANIMALS_ENABLED:
        fert_b = fert_buy_plan(farm_state, market_state, prices, params,
                               money_proj, slots)
        orders.extend(fert_b)
        slots -= len(fert_b)
        money_proj -= sum(o[2] for o in fert_b) * prices.get("FERTILIZER", 100)
        wheat_b = wheat_buy_plan(farm_state, market_state, prices,
                                 money_proj, slots)
        orders.extend(wheat_b)
        slots -= len(wheat_b)
        money_proj -= sum(o[2] for o in wheat_b) * prices.get("WHEAT", 25)

    if slots < 0:
        orders = orders[:MAX_ORDERS]

    # ---- unit scheduling ------------------------------------------------
    tasks, plant_map, build_map = build_tasks(farm, farm_state, plan,
                                              market_state, day, prices)
    fa, hand_actions = schedule_units(farm, farm_state, market_state, plan,
                                      tasks, plant_map, build_map, day, prices)

    return {"farmer": fa, "hands": hand_actions, "market": orders}

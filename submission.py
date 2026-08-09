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

HARD_CAPS = {"WHEAT": 60, "CARROT": 0, "TOMATO": 10,
             "STRAWBERRY": 16, "MELON": 16}

ENDGAME_DAY   = 27
SHED_FORCE    = 86
ACTIONS_PER_HAND = 14
HANDS_MAX     = 7

# Realistic daily actions one hired hand performs once travel and queuing
# are accounted for, as a fraction of the theoretical ceiling.
HAND_EFFICIENCY  = 0.55
HAND_VALUE_MIN  = 6
HIRE_HOUR_CUT = 12

LAND_DAYS     = [99, 99, 99]
LAND_CASH     = [1000, 2000, 4000]
LAND_LAST_DAY = 0

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


def _next_prod_day(crop, planted_day, day):
    """Next production day strictly after `day`, or None when the plant is
    done.  Production happens at the day-rollover whose day equals
    planted_day + FIRST_YIELD + k*INTERVAL."""
    for k in range(MAX_YIELD[crop]):
        nd = planted_day + FIRST_YIELD[crop] + k * INTERVAL[crop]
        if nd > day and nd <= LAST_HARVEST:
            return nd
    return None


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


# ============================================================================
# Adaptive opponent observation (exact market-ledger inference)
# ============================================================================
_OPP_OBS = {"day": -1, "mkt_prev": None, "our_sold": None, "our_bought": None,
            "rates": {}}


def _town_units(day, product, town):
    """Exact town consumption of `product` on `day` (units), known shops only:
    town center drains 2/day x demand-multiplier; each unlocked shop drains
    6/day per demanded product (12 for single-product shops)."""
    mult = 8 if day >= 20 else (4 if day >= 10 else 2)
    units = float(mult * 2)
    for s in town.get("unlocked_shops", []) or []:
        if product in SHOPS[s]:
            units += 24.0 / 4.0 * (2 if len(SHOPS[s]) == 1 else 1)
    return units


def observe_opponent(day, hour, mkt_inv, orders, town):
    """Track the public market ledger day over day and infer the opponent's
    actual sales per product (units/day, EMA-smoothed).

    The ledger is exactly: inventory change = our sells - our product buys
    - opponent sells + their product buys + town drain.  Their BUY_PRODUCT
    (wheat/fertilizer) is small and unobserved; the clamp absorbs it.
    """
    o = _OPP_OBS
    if o["mkt_prev"] is None:
        o["day"] = day
        o["mkt_prev"] = dict(mkt_inv)
        o["our_sold"] = {p: 0 for p in PRODUCTS}
        o["our_bought"] = {p: 0 for p in PRODUCTS}
        return
    if day == o["day"]:
        for ord_ in orders or []:
            if len(ord_) < 3:
                continue
            op = ord_[0]
            if op == "SELL" and ord_[1] in o["our_sold"]:
                o["our_sold"][ord_[1]] += int(ord_[2])
            elif op == "BUY_PRODUCT" and ord_[1] in o["our_bought"]:
                o["our_bought"][ord_[1]] += int(ord_[2])
        return
    if hour != 0:
        # incomplete day: restart the ledger window on the new day
        o["day"] = day
        o["mkt_prev"] = dict(mkt_inv)
        o["our_sold"] = {p: 0 for p in PRODUCTS}
        o["our_bought"] = {p: 0 for p in PRODUCTS}
        return
    prev_day = o["day"]
    for p in PRODUCTS:
        if p == "FERTILIZER":
            continue
        prev = o["mkt_prev"].get(p)
        cur = mkt_inv.get(p)
        if prev is None or cur is None:
            continue
        drained = _town_units(prev_day, p, town)
        opp_sold = prev - cur + o["our_sold"].get(p, 0) \
            - o["our_bought"].get(p, 0) + drained
        opp_sold = max(0.0, min(opp_sold, 150.0))
        old = o["rates"].get(p, 0.0)
        o["rates"][p] = 0.6 * old + 0.4 * opp_sold
    o["day"] = day
    o["mkt_prev"] = dict(mkt_inv)
    o["our_sold"] = {p: 0 for p in PRODUCTS}
    o["our_bought"] = {p: 0 for p in PRODUCTS}


def apply_observed_opponent(opp, day, hour, mkt_inv, orders, town):
    """Blend the biological opponent estimate with the observed market
    inference (observed sales dominate once the ledger has signal)."""
    observe_opponent(day, hour, mkt_inv, orders, town)
    rates = _OPP_OBS["rates"]
    if not rates:
        return
    for p, r in rates.items():
        if r > 0:
            bio = opp["daily_rate"].get(p, 0.0)
            opp["daily_rate"][p] = max(bio, r)


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


# Average daily labor demand of a tile through its lifetime: planting,
# in-window watering (every other day), harvesting, fertilizing.  Melons are
# cheap to tend, strawberries expensive (fertilizer renewals), wheat/carrot
# are one-time bursts.
_CROP_LABOR = {"WHEAT": 1.5, "CARROT": 1.5, "TOMATO": 1.25,
               "STRAWBERRY": 1.5, "MELON": 0.85}
_LABOR_PER_TILE = 2.0    # planning divisor: weighted average of the above


def _tile_demand(t, day):
    """Approximate daily actions a plant needs from here on.  Ongoing
    crops demand water+harvest while productions remain; one-time
    crops demand water only inside their bonus window and then just a
    harvest."""
    c = t["crop"]
    if ONGOING[c]:
        if _productions_remaining(c, t["planted_day"], day) > 0:
            return _CROP_LABOR[c]
        return 0.3                     # spent: just a DIG
    if day >= MAX_YIELD_DAY[c]:
        return 0.8                     # matured: harvest only
    return _CROP_LABOR[c]              # growing phase


def crop_plan(farm_state, market_state, opp, day, prices, inv, params, money,
              land_extra=0):
    tiles_avail = farm_state["empty"] + farm_state["weeds"] + len(farm_state["spent"]) \
        + land_extra
    market_state.setdefault("harvest_price",
                            {c: prices.get(c, BASE_PRICE[c]) for c in CROPS_LIST})
    if tiles_avail <= 0:
        # The allocator reads these keys unconditionally; a full field must
        # still report its labor picture (the relief-hire decision depends
        # on it), so include them in the early return too.
        n_units = farm_state.get("n_units", 1)
        capacity = n_units * ACTIONS_PER_HAND * HAND_EFFICIENCY
        demand_now = 5.0
        for (x, y, t) in farm_state["plants"]:
            demand_now += _tile_demand(t, day)
        demand_now += farm_state["n_animals"] * 3.0
        return {"crops": {}, "ranked": [], "profit": {}, "fert_reserve": 0,
                "tiles_avail": 0, "labor_spare": capacity - demand_now,
                "labor_capacity": capacity}

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
    capacity = n_units * ACTIONS_PER_HAND * HAND_EFFICIENCY
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
                      max(0, int((labor_spare - mandatory_reserve + 8.0) / _LABOR_PER_TILE)))

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
def sell_batch(item, qty, mkt_inv, params):
    """Sequential one-unit-at-a-time sale (mirrors the env's per-unit
    lockstep pricing): returns (revenue, units added to market supply).
    Sales at $1 do not add to market inventory."""
    total = 0.0
    added = 0
    inv = mkt_inv
    for _ in range(int(qty)):
        pr = price_at(item, inv, params)
        total += pr
        if pr > 1:
            inv += 1
            added += 1
    return total, added


def _policy_sell_qty(p, day, qty, mkt_inv, params, drain_rate, opp_rate,
                     our_rate, forced):
    """SHARED sell policy - the real sell_plan AND the simulator use the
    same rule, so projected prices stay consistent with play.

    The market absorbs any quantity instantly at the current price, but
    every sale unit raises market inventory and drags the price down.  So
    the rule is:
      - forced (endgame / shed pressure / cash crunch) or dead price: sell
        everything ($1 sales add no inventory, so dumping is free);
      - price at/above base: sell the largest batch whose AVERAGE price
        stays >= 0.8x the current price (bulk harvests clear quickly);
      - price below base (glut building): trickle at the drain rate so we
        stop shovelling supply into a falling market.
    """
    if qty <= 0:
        return 0.0
    price = price_at(p, mkt_inv, params)
    base = params[p]["base"]
    if forced or price <= 1:
        return qty
    n = _safe_batch(p, qty, mkt_inv, params, price)
    if n < 1:
        return 1.0 if qty >= 1 else 0.0
    if price < base:
        n = min(n, max(1.0, drain_rate - opp_rate) + 2.0)
    return min(qty, max(1.0, n))


def build_sim(farm_state, market_state, day, prices, inv, params, money, opp):
    """Assemble the deterministic season-projection state:

        cash / market inventory / our inventory
        + per-day production schedules (ours, from actual tiles/animals)
        + expected opponent sales per day
        + expected town drain per day

    The schedule carries units as of the day they become harvestable, so
    production -> our inventory -> sell decision -> market inventory ->
    price -> cash is a real causal chain, not production == cash.
    """
    prod = {p: [0.0] * 30 for p in PRODUCTS}
    for (x, y, t) in farm_state["plants"]:
        c = t["crop"]
        if ONGOING[c]:
            for k in range(MAX_YIELD[c]):
                nd = t["planted_day"] + FIRST_YIELD[c] + k * INTERVAL[c]
                if day < nd <= LAST_HARVEST and nd < 30:
                    bonus = 1.0 if t.get("fertilized_until_day", -1) >= nd else 0.0
                    prod[c][nd] += 1.0 + bonus
        else:
            rem = max(0, t.get("yield_units", 0))
            age = day - t.get("planted_day", day)
            if rem > 0 and age >= MAX_YIELD_DAY[c]:
                prod[c][day] += rem
            elif rem > 0:
                start = (MAX_YIELD_DAY[c] + 1) // 2
                total = rem
                for wd in range(max(start, day + 1), MAX_YIELD_DAY[c] + 1):
                    if wd <= LAST_HARVEST:
                        total += 2.0 if t.get("fertilized_until_day", -1) >= wd else 1.0
                hd = min(29, day + (MAX_YIELD_DAY[c] - age))
                prod[c][hd] += min(float(MAX_YIELD[c]), total)
    cp = care_probability(farm_state, day)
    for (x, y, t) in farm_state["animal_tiles"]:
        a = t["animal"]
        p = ANIMAL_PRODUCT[a]
        rem = max(0, t.get("yield_units", 0))
        if rem > 0:
            prod[p][day] += rem
        k = 0
        while True:
            nd = t.get("placed_day", day) + ANIMAL_FIRST[a] + k * ANIMAL_INTERVAL[a]
            if nd > LAST_HARVEST or nd >= 30:
                break
            if nd > day:
                prod[p][nd] += 1.0 + cp
            k += 1
    shed_inv = {p: float(market_state["shed"].get(p, 0)) for p in PRODUCTS}
    for iv in market_state["inventories"]:
        for p, n in iv.items():
            if p in shed_inv:
                shed_inv[p] += n
    town = market_state["town"]
    drain_day = {p: [town_rate(d, p, town) for d in range(30)] for p in PRODUCTS}
    drain_avg = {p: sum(drain_day[p][d] for d in range(max(day, 1), 30))
                 / max(1, 30 - max(day, 1)) for p in PRODUCTS}
    return {
        "day": day,
        "cash": float(money),
        "mkt": {p: float(inv.get(p, params[p]["I0"])) for p in PRODUCTS},
        "inv": shed_inv,
        "prod": prod,
        "opp": {p: opp["daily_rate"].get(p, 0.0) for p in PRODUCTS},
        "drain_day": drain_day,
        "drain_avg": drain_avg,
        "animals": farm_state["n_animals"],
        "town": town,
        "liquidate_from": ENDGAME_DAY,
        "care_p": cp,
    }


def run_sim(sim, params):
    """Deterministic projection of FINAL BANK CASH under the shared sell
    policy.  Pure (does not mutate `sim`; result cached on it).  One day of
    simulation: production arrives, wheat feed is consumed (shortfalls are
    bought at the market), opponent sales hit the market, we sell per the
    policy, town drains, prices move."""
    cached = sim.get("_v")
    if cached is not None:
        return cached
    mkt = dict(sim["mkt"])
    inv = dict(sim["inv"])
    cash = float(sim["cash"])
    n_anim = int(sim.get("animals", 0))
    liquidate_from = int(sim.get("liquidate_from", ENDGAME_DAY))
    shed_cap = 100.0
    drain_day = sim["drain_day"]
    drain_avg = sim["drain_avg"]
    prod = sim["prod"]
    opp = sim["opp"]
    for d in range(int(sim["day"]), 30):
        for p in prod:
            v = prod[p][d]
            if v > 0:
                inv[p] = inv.get(p, 0.0) + v
        total = sum(inv.values())
        if total > shed_cap:
            # mirror the env's end-of-day overflow discard
            excess = total - shed_cap
            for p in sorted(inv, key=lambda k: params[k]["base"]):
                if excess <= 0:
                    break
                take = min(inv[p], excess)
                inv[p] -= take
                excess -= take
        if n_anim > 0:
            have = inv.get("WHEAT", 0.0)
            if have >= n_anim:
                inv["WHEAT"] = have - n_anim
            else:
                short = n_anim - have
                inv["WHEAT"] = 0.0
                pb = price_at("WHEAT", max(0, mkt.get("WHEAT", 10000) - 1), params)
                cash -= short * pb
                mkt["WHEAT"] = max(0, mkt["WHEAT"] - short)
        for p in opp:
            r = opp[p]
            if r > 0:
                mkt[p] = mkt.get(p, params[p]["I0"]) + r
        forced = d >= liquidate_from or sum(inv.values()) >= 86 or cash < 150
        for p in SELL_ORDER:
            if p == "FERTILIZER":
                continue
            q = inv.get(p, 0.0)
            if q <= 0:
                continue
            qty = _policy_sell_qty(p, d, q, mkt[p], params, drain_avg[p],
                                   opp.get(p, 0.0), prod[p][d], forced)
            if qty >= 1:
                rev, added = sell_batch(p, qty, mkt[p], params)
                cash += rev
                mkt[p] += added
                inv[p] -= qty
        for p in mkt:
            mkt[p] = max(0.0, mkt[p] - drain_day[p][d])
    for p in SELL_ORDER:
        if p == "FERTILIZER":
            continue
        q = inv.get(p, 0.0)
        if q > 0:
            rev, _ = sell_batch(p, q, mkt[p], params)
            cash += rev
            inv[p] = 0.0
    sim["_v"] = cash
    return cash


def _safe_batch(item, q, inv, params, price):
    """Largest batch to sell this turn without dragging the average price
    below 0.8x the current price."""
    n = 1
    while n < q and batch_avg(item, n, inv, params) >= 0.8 * price:
        n += 1
    return max(1, n - 1)


def sell_plan(farm_state, market_state, opp, day, prices, inv, params):
    """Real-time sell orders under the SHARED sell policy (the same rule the
    simulator projects with), so actual sales match the projected market.
    Batch-protected and slot-budgeted."""
    orders = []
    shed = market_state["shed"]
    slots = MAX_ORDERS
    shed_total = sum(shed.values())
    money = market_state["money"]
    endgame = day >= ENDGAME_DAY
    forced = shed_total >= SHED_FORCE or endgame or money < 150
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
        mkt_inv = inv.get(p, params[p]["I0"])
        drain = drain_between(day, p, days_left, market_state["town"]) / days_left
        opp_sell = opp["daily_rate"].get(p, 0.0)
        our_rate = my_supply_rates(farm_state, day).get(p, 0.0)
        qty = _policy_sell_qty(p, day, q, mkt_inv, params, drain, opp_sell,
                               our_rate, forced)
        n = int(qty)
        n = min(n, _safe_batch(p, q, mkt_inv, params, prices.get(p, BASE_PRICE[p])))
        if n >= 1:
            orders.append(["SELL", p, n])
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
        # plan["crops"] is the ABSOLUTE field composition, so the seed
        # deficit is target minus what is already planted or in the shed.
        # (An incremental plan made this go negative once a crop was on the
        # field, freezing the mid-game strawberry pipeline.)
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


def _sim_clone(sim):
    s = dict(sim)
    s["prod"] = {p: list(sim["prod"][p]) for p in sim["prod"]}
    s["mkt"] = dict(sim["mkt"])
    s["inv"] = dict(sim["inv"])
    s.pop("_v", None)
    return s


def _crop_unit_cost(c, day, prices, params, fert_ok):
    """Cash cost of one tile of `crop` planted today (seeds + fertilizer)."""
    cost = SEED_COST[c]
    if fert_ok.get(c):
        n_fert = 3 if c == "STRAWBERRY" else 1
        cost += n_fert * prices.get("FERTILIZER", BASE_PRICE["FERTILIZER"])
    return cost


def _apply_crop(sim, c, day, prices, params, fert_ok, k=1):
    """Add `k` tiles of crop `c` (planted today) to the sim: deduct seed +
    fertilizer cash, add the plant's production schedule."""
    if not can_finish(c, day):
        return None
    s = _sim_clone(sim)
    s["cash"] -= k * _crop_unit_cost(c, day, prices, params, fert_ok)
    prod = s["prod"][c]
    if ONGOING[c]:
        for i in range(k):
            n = 0
            for j in range(MAX_YIELD[c]):
                nd = day + FIRST_YIELD[c] + j * INTERVAL[c]
                if day < nd <= LAST_HARVEST and nd < 30:
                    prod[nd] += 1.0 + (1.0 if (fert_ok.get(c) and n < 3) else 0.0)
                    n += 1
    else:
        start = (MAX_YIELD_DAY[c] + 1) // 2
        total = 1.0
        for wd in range(max(start, day + 1), MAX_YIELD_DAY[c] + 1):
            total += 2.0 if fert_ok.get(c) else 1.0
        total = min(float(MAX_YIELD[c]), total)
        hd = day + MAX_YIELD_DAY[c]
        if hd <= LAST_HARVEST and hd < 30:
            prod[hd] += total * k
    return s


def _apply_animal(sim, a, day):
    """Add one `a` bought today (placed immediately): cash cost + production
    schedule at (1 + expected CARE) per tick + one more wheat/day to feed."""
    s = _sim_clone(sim)
    s["cash"] -= ANIMAL_COST[a]
    s["animals"] = int(s.get("animals", 0)) + 1
    p = ANIMAL_PRODUCT[a]
    prod = s["prod"][p]
    cp = s.get("care_p", 1.0)
    k = 0
    while True:
        nd = day + 1 + ANIMAL_FIRST[a] + k * ANIMAL_INTERVAL[a]
        if nd > LAST_HARVEST or nd >= 30:
            break
        prod[nd] += 1.0 + cp
        k += 1
    return s


def _apply_land(sim, day, prices, params, ranked, fert_ok, n_tiles=25):
    """Add a new quadrant: fill it with the best-ranked crop mix."""
    s = _sim_clone(sim)
    remaining = n_tiles
    for c in ranked:
        if remaining <= 0:
            break
        take = min(HARD_CAPS.get(c, 99), remaining)
        if take <= 0 or not can_finish(c, day):
            continue
        sc = _apply_crop(s, c, day, prices, params, fert_ok, take)
        if sc is None:
            continue
        s = sc
        remaining -= take
    if remaining > 0:
        return None
    return s


def capital_allocator(farm_state, market_state, opp, day, prices, inv, params,
                      money, tiles_avail, labor_spare, labor_capacity,
                      ranked, hard_start):
    """GREEDY MARGINAL CAPITAL ALLOCATION.

    Every candidate (one tile of each crop, one animal of each type, one
    quadrant, one hired hand) is valued as the expected incremental final
    bank cash produced by running the season simulator with and without the
    investment.  The highest positive delta/cost is applied and the search
    repeats until no candidate is profitable.
    """
    sim = build_sim(farm_state, market_state, day, prices, inv, params,
                    money, opp)
    fert_ok = {c: _fert_marginal(c, day, prices, params, market_state) > 0
               for c in CROPS_LIST}
    # plan["crops"] is the ABSOLUTE field composition the allocator wants
    # (what is planted now + what it commits to plant today).  v5.2's plan
    # had this property, and the seed buyers and the task scheduler both
    # consume it that way: seed_plan buys the difference vs (seeds + planted)
    # and build_tasks plants `remaining` tiles of each crop on empty cells.
    # An incremental plan (only today's commits) made seed_plan's deficit go
    # negative once a crop was already on the field, which froze the entire
    # mid-game strawberry pipeline (the day-6..12 gap behind v5.2).
    if day not in (0, 1, 13):
        hard_start = {}
    base = dict(hard_start)
    for c in CROPS_LIST:
        if c not in base:
            base[c] = farm_state["crop_counts"].get(c, 0)
    plan = {"crops": dict(base), "ranked": list(ranked), "profit": {},
            "fert_reserve": 0, "tiles_avail": tiles_avail,
            "labor_spare": labor_spare, "labor_capacity": labor_capacity,
            "animals": {}, "land": False, "hires": 0}
    # Build-out days spend the bankroll down to the safety floor.  v5.2's
    # proven flow kept planting (mostly strawberry seeds at $100) even with
    # ~$250 in the wallet, so no extra working-cash buffer is subtracted
    # here; the sell policy and harvest cash flow cover mid-season bills.
    budget = max(0, money - CASH_FLOOR)
    tiles_left = max(0, tiles_avail - sum(hard_start.values()))
    caps_left = {c: max(0, HARD_CAPS[c] - base.get(c, 0))
                 for c in CROPS_LIST}
    # Labor already promised by the plan; animals and extra tiles must fit in
    # what remains of the crew's daily action budget (3.5 actions per tile,
    # 3 per animal).
    labor_committed = 0.0
    run_sim(sim, params)

    # Hands are the cheapest investment in the game (FIB wages) and they
    # unlock the labor the tile plan needs, so they compete in the same
    # pool as every other candidate.  Each hand services ~6 tiles.
    n_extra = max(0, len(market_state["unlocked"]) - 1)
    hcost = FIB[min(farm_state["hires_today"], len(FIB) - 1)]

    for _ in range(12):
        # Stop planting only when the field is full AND the crew is
        # comfortable; a full field still needs the relief-hire candidate
        # below (hired hands are valued by the plants they keep alive, so
        # they must be evaluated even when there is nothing left to plant).
        if tiles_left <= 0 and plan["hires"] >= farm_state["hires_today"] \
                and (labor_spare is None or labor_spare >= 8.0):
            break
        # ---- one pool: crops (1-tile probes), animals, land, hands --------
        best_c, best_d, best_uc = None, 0.0, 0.0
        for c in CROPS_LIST:
            if not can_finish(c, day) or caps_left[c] <= 0:
                continue
            # Melons are a day-0 crop: planted later they pay 5 units into
            # the day-12 glut, and the sim's late-melon delta (6 units at
            # day+12) outranks strawberries in this pool, stealing the
            # daily budget that should go to the mid-game strawberry wave.
            if c == "MELON" and day >= 6:
                continue
            # The budget gate is the SEED cash only: fertilizer is bought
            # separately (fert_buy_plan, money >= 400) and the sim's delta
            # already includes the bonus units it enables.  Charging the
            # whole seed+fert bundle here froze all planting while the
            # wallet was below ~$700, leaving tiles empty for days.
            uc_seed = SEED_COST[c]
            if uc_seed > budget:
                continue
            uc = _crop_unit_cost(c, day, prices, params, fert_ok)
            sc = _apply_crop(sim, c, day, prices, params, fert_ok, 1)
            if sc is None:
                continue
            d = run_sim(sc, params) - run_sim(sim, params)
            if d > best_d:
                best_d, best_c, best_uc = d, c, uc_seed
        candidates = [(best_d, best_uc, "crop", best_c)] if best_c and best_d > 0 else []

        placement_room = farm_state["empty_coop"] + farm_state["empty_pasture"] \
            + farm_state["empty"]
        pending_animals = sum(market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
        feed_cap = farm_state["crop_counts"]["WHEAT"] * 1.2 \
            + market_state["shed"].get("WHEAT", 0) / 8.0 \
            + max(0, money - 400) / 400.0
        feed_cap = max(2.0, min(feed_cap, 16.0))
        for a in ANIMALS_LIST:
            have = farm_state["animal_counts"][a] + market_state["shed"].get(a, 0)
            if have >= min(ANIMAL_CAPS[a], int(feed_cap)):
                continue
            if day + ANIMAL_FIRST[a] > LAST_HARVEST or day + 1 + ANIMAL_FIRST[a] > 28:
                continue
            if _animal_prods_left(a, day + 1, day) <= 1:
                continue
            if farm_state["n_animals"] + pending_animals + 1 > placement_room:
                continue
            # Animals only after the crop field is saturated: every animal
            # steals ~3 daily actions from watering/harvesting that the early
            # crew cannot spare while the field is still being planted.
            n_animals_plan = farm_state["n_animals"] + sum(plan["animals"].values())
            if tiles_left > 0 or labor_spare is None or \
                    labor_spare - labor_committed < n_animals_plan * 3 + 12:
                continue
            if budget < ANIMAL_COST[a] + 300:
                continue
            sa = _apply_animal(sim, a, day)
            da = run_sim(sa, params) - run_sim(sim, params)
            if da > 0:
                candidates.append((da, ANIMAL_COST[a], "animal", a))
        # Land only pays if the quadrant can still host a full-wave crop:
        # tiles bought past the best crop's last production day return less
        # than their price.  v5.2 never bought land because its forward
        # valuation refused it; the same gate keeps the sim's optimistic
        # marginal deltas from buying a third quadrant at day 15.
        land_ok = False
        for c in plan["ranked"]:
            if can_finish(c, day) and day + FIRST_YIELD[c] \
                    + (MAX_YIELD[c] - 1) * INTERVAL[c] <= 29:
                land_ok = True
                break
        if land_ok and n_extra < 3 and LAND_DAYS[n_extra] <= day <= LAND_LAST_DAY \
                and budget >= LAND_CASH[n_extra] and labor_spare is not None \
                and labor_spare >= 20:
            used = sum(farm_state["crop_counts"].values()) + farm_state["n_animals"] \
                + farm_state["empty_coop"] + farm_state["empty_pasture"]
            usable = 25 * len(market_state["unlocked"]) - 4
            if used >= 0.6 * usable:
                sl = _apply_land(sim, day, prices, params, plan["ranked"], fert_ok)
                if sl is not None:
                    dl = run_sim(sl, params) - run_sim(sim, params)
                    if dl > 0:
                        candidates.append((dl, LAND_CASH[n_extra], "land", None))
        # Hands are paid daily wages (FIB 10..50) and work the SAME day, so
        # they must be affordable from the cash floor, not from the crop
        # budget (which covers one-shot seed purchases).
        hire_cash = money - CASH_FLOOR
        if day <= 28 and hire_cash >= hcost \
                and plan["hires"] < HANDS_MAX \
                and farm_state["hires_today"] + plan["hires"] < HANDS_MAX:
            # value of a hand = the best tile delta it can serve (tiles it
            # unlocks to plant today) OR the labor relief it brings to the
            # existing field (a plant becomes a weed after two unwatered
            # days, so when the crew is underwater a hand is worth the crops
            # it saves).  Hired hands work the same day and are paid daily.
            tiles_served = 0
            if best_c and tiles_left > 0:
                tiles_served = min(int(ACTIONS_PER_HAND * HAND_EFFICIENCY
                                       / _LABOR_PER_TILE), tiles_left)
            relief = 0.0
            if labor_spare is not None:
                relief = max(0.0, min(3.0, (12.0 - labor_spare) / _LABOR_PER_TILE))
            dh = 0.0
            if tiles_served >= 1 and best_c:
                sh = _sim_clone(sim)
                sh["cash"] -= hcost
                sh2 = _apply_crop(sh, best_c, day, prices, params, fert_ok,
                                  min(tiles_served,
                                      int(hire_cash // max(1, best_uc))))
                if sh2 is not None:
                    dh = run_sim(sh2, params) - run_sim(sim, params)
            if dh <= 0.0 and relief >= 0.8:
                # No new crop is worth planting today (the field is full or
                # the market is glutted): the hand is still worth the plants
                # it keeps alive.  Value one tile of protection at the best
                # marginal tile value we know, or a conservative 150$ for the
                # standing crop.
                per_tile = best_d if best_c else 150.0
                dh = max(tiles_served, relief) * per_tile * 0.6
            if dh > 0:
                candidates.append((dh, hcost, "hire", None))
        if not candidates:
            break
        candidates.sort(key=lambda t: -(t[0] / max(1.0, t[1])))
        d_best, cost_best, kind, arg = candidates[0]
        if d_best <= 0:
            break
        if kind == "crop":
            k = min(caps_left[arg], tiles_left, 4, max(1, int(budget // cost_best)))
            if k <= 0:
                break
            sc = _apply_crop(sim, arg, day, prices, params, fert_ok, k)
            if sc is None:
                break
            plan["crops"][arg] = plan["crops"].get(arg, 0) + k
            plan["profit"][arg] = d_best
            sim = sc
            budget -= k * cost_best
            tiles_left -= k
            caps_left[arg] -= k
            labor_committed += k * _LABOR_PER_TILE
        elif kind == "animal":
            sim = _apply_animal(sim, arg, day)
            budget -= cost_best
            plan["animals"][arg] = plan["animals"].get(arg, 0) + 1
            farm_state["n_animals"] += 1
            labor_committed += 3.0
        elif kind == "land":
            sim = _apply_land(sim, day, prices, params, plan["ranked"], fert_ok)
            budget -= cost_best
            plan["land"] = True
            tiles_left += 25
            caps_left = {c: HARD_CAPS[c] - plan["crops"].get(c, 0) for c in CROPS_LIST}
            labor_committed += 25 * _LABOR_PER_TILE
        elif kind == "hire":
            sim = _sim_clone(sim)
            sim["cash"] -= cost_best
            budget -= cost_best
            plan["hires"] += 1

    # ---- final ranking: per-tile sim deltas (drives the scheduler) --------
    for c in CROPS_LIST:
        if not can_finish(c, day):
            continue
        if SEED_COST[c] > budget and budget > 0:
            continue
        sc = _apply_crop(sim, c, day, prices, params, fert_ok, 1)
        if sc is None:
            continue
        d = run_sim(sc, params) - run_sim(sim, params)
        plan["profit"][c] = max(0.0, d)
    plan["ranked"] = [c for c in CROPS_LIST if plan["profit"].get(c, 0) > 0]
    plan["ranked"].sort(key=lambda c: plan["profit"][c], reverse=True)

    # ---- fertilizer reserve for the planned mix ---------------------------
    fert_reserve = 0
    for c, n in plan["crops"].items():
        if c == "STRAWBERRY" and fert_ok[c]:
            fert_reserve += 3 * n
        elif c == "TOMATO" and fert_ok[c]:
            fert_reserve += 1 * n
    for (x, y, t) in farm_state["plants"]:
        c = t["crop"]
        if c == "STRAWBERRY" and fert_ok[c] \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 2
        elif c == "TOMATO" and fert_ok[c] \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 1
    plan["fert_reserve"] = min(fert_reserve, FERT_USE_MAX * 2)
    return plan


# ============================================================================
# Order builders (the allocator decides, these just execute)
# ============================================================================
def hire_orders(plan, farm_state, money, slots, day, hour):
    orders = []
    if day >= 29 or hour >= HIRE_HOUR_CUT or slots <= 0:
        return orders
    target = plan.get("hires", 0)
    hires_today = farm_state["hires_today"]
    while hires_today < target and slots > 0:
        cost = FIB[min(hires_today, len(FIB) - 1)]
        if money - cost < CASH_FLOOR:
            break
        orders.append(["HIRE"])
        slots -= 1
        hires_today += 1
        money -= cost
    return orders


def animal_orders(plan, farm_state, market_state, money, slots):
    orders = []
    if slots <= 0:
        return orders
    placement_room = farm_state["empty_coop"] + farm_state["empty_pasture"] \
        + farm_state["empty"]
    pending = sum(market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
    for a, n in plan.get("animals", {}).items():
        for _ in range(n):
            if slots <= 0 or money < ANIMAL_COST[a] + CASH_FLOOR + 300:
                break
            if farm_state["n_animals"] + pending >= placement_room:
                break
            orders.append(["BUY_ANIMAL", a, 1])
            slots -= 1
            money -= ANIMAL_COST[a]
            pending += 1
    return orders


_STRAT = {"day": -1, "plan": None, "money": None, "shed_total": None,
          "mkt_total": None, "unlocked": None, "shops": None, "prices": None}
_LAND_BOUGHT_DAY = -1


def _should_replan(day, money, shed, mkt_inv, unlocked, shops, prices):
    s = _STRAT
    if s.get("day") != day:
        return True
    if abs(money - (s.get("money") or -1e18)) > 250:
        return True
    if abs(sum(shed.values()) - (s.get("shed_total") or -1e18)) > 20:
        return True
    if abs(sum(mkt_inv.values()) - (s.get("mkt_total") or -1e18)) > 80:
        return True
    if s.get("unlocked") != tuple(sorted(unlocked)):
        return True
    if s.get("shops") != tuple(sorted(shops)):
        return True
    old = s.get("prices") or {}
    for p in prices:
        o = old.get(p)
        if o is not None and abs(prices[p] - o) > 0.25 * max(1, o):
            return True
    return False


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
    # Daily planting cap: planting too many tiles in one burst starves the
    # same crew's watering rounds (every plant becomes a weed after two
    # unwatered days).  v5.2's proven pace was ~8-11 tiles/day; unplanted
    # tiles stay empty and the plan's plant_map refills next day.
    plant_cap = max(4, int(farm_state.get("n_units", 1) * ACTIONS_PER_HAND
                           * HAND_EFFICIENCY * 0.22))
    for (x, y) in empty_cells:
        if plant_cap <= 0:
            break
        for c in plan_crops:
            if remaining.get(c, 0) > 0 and seed_budget.get(c, 0) > 0:
                plant_map[(x, y)] = c
                remaining[c] -= 1
                seed_budget[c] -= 1
                plant_cap -= 1
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
                            and t.get("fertilized_until_day", -1) < day \
                            and _next_prod_day(crop, t.get("planted_day", day), day) \
                            is not None \
                            and _next_prod_day(crop, t.get("planted_day", day), day) \
                            <= day + 3:
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

    global _LAND_BOUGHT_DAY
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

    if hour == 0 and day == 0:
        _STRAT.update({"day": -1, "plan": None, "money": None,
                       "shed_total": None, "mkt_total": None,
                       "unlocked": None, "shops": None, "prices": None})
        _OPP_OBS.update({"day": -1, "mkt_prev": None, "our_sold": None,
                         "our_bought": None, "rates": {}})
        _LAND_BOUGHT_DAY = -1

    # ---- strategic re-plan: only at day boundaries.  Intra-day replans
    # oscillate: money moves as buy orders process, so a money/price trigger
    # re-plans every few turns and the plan shrinks (hires and crops get
    # dropped mid-day, seeds stop, watering falls behind).  Decisions made at
    # hour 0 with the cached plan stay coherent for the whole day.  During
    # the build-out phase (day <= 6) we re-plan every morning: the crew
    # grows daily and idle cash must be converted into crops, which the
    # money/shed-change triggers below would otherwise miss while the field
    # sits half empty.
    replan = hour == 0 and (day <= 12 or _should_replan(day, money, shed, inv,
                                                        unlocked,
                                                        town.get("unlocked_shops", []),
                                                        prices))
    if replan:
        crop_feas = crop_plan(farm_state, market_state, opp, day, prices, inv,
                              params, money)
        hard_start = {}
        if day <= 1:
            # Day-0 mix: 16 melons (the day-13 cash engine) + 8 wheat.  The
            # wheat is a short bridge: it pays out day 4-5 and its tiles
            # recycle into strawberries on days 5-8, which produce 6-7 full
            # waves at the season's highest prices.  Day-0 strawberries
            # (which the marginal sim prefers) yield only 4 waves at lower
            # prices and occupy the tiles that should host the mid-game
            # crop.  This replicates v5.2's proven sequence.
            hard_start = {"MELON": 16, "WHEAT": 8}
        elif day == 13:
            # Day-13 rebuild: strawberries planted today still finish 4 full
            # waves (23, 25, 27, 29), and the sim's marginal delta collapses
            # as tiles pile up (its glut model over-penalizes the day-23-29
            # waves), which stalls the rebuild on some seeds.  v5.2 finished
            # this in two days with a full strawberry field; force the same
            # here and let the allocator's deltas choose the tomato/wheat
            # balance on the tiles that remain.
            hard_start = {}
        try:
            plan = capital_allocator(farm_state, market_state, opp, day,
                                     prices, inv, params, money,
                                     crop_feas["tiles_avail"],
                                     crop_feas["labor_spare"],
                                     crop_feas["labor_capacity"],
                                     crop_feas["ranked"], hard_start)
        except Exception:
            plan = crop_feas
        _STRAT.update({"day": day, "plan": plan, "money": money,
                       "shed_total": sum(shed.values()),
                       "mkt_total": sum(inv.values()),
                       "unlocked": tuple(sorted(unlocked)),
                       "shops": tuple(sorted(town.get("unlocked_shops", []))),
                       "prices": dict(prices)})
    else:
        plan = _STRAT.get("plan")
        if plan is None:
            plan = crop_plan(farm_state, market_state, opp, day, prices, inv,
                             params, money)
    market_state["fert_reserve"] = plan.get("fert_reserve", 0)

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
    apply_observed_opponent(opp, day, hour, inv, sell_orders, town)

    # ---- hire first: hands are the cheapest labor in the game --------------
    hire_ord = hire_orders(plan, farm_state, money_proj, slots, day, hour)
    orders.extend(hire_ord)
    slots -= len(hire_ord)
    money_proj -= sum(FIB[min(farm_state["hires_today"] + i - 1, len(FIB) - 1)]
                      for i in range(1, len(hire_ord) + 1))

    # ---- land: buying land expands the crop plan for today -----------------
    # At most one land per day: the plan's land flag stays true all day, and
    # re-firing the order every hour while money allows burned 1000/2000/4000
    # in a single day (seed 1 regressed ~7K).
    n_extra = max(0, len(market_state["unlocked"]) - 1)
    if plan.get("land") and n_extra < len(LAND_CASH) \
            and _LAND_BOUGHT_DAY != day:
        if slots > 0 and money_proj >= LAND_CASH[n_extra]:
            orders.append(["BUY_LAND"])
            _LAND_BOUGHT_DAY = day
            slots -= 1
            money_proj -= LAND_CASH[n_extra]

    # ---- animals rank ahead of seeds: their ROI per order slot is higher
    # and they are capped, so they must not starve when slots are tight -----
    if ANIMALS_ENABLED:
        animal_b = animal_orders(plan, farm_state, market_state, money_proj,
                                 slots)
        orders.extend(animal_b)
        slots -= len(animal_b)
        money_proj -= sum(ANIMAL_COST[o[1]] for o in animal_b)

    seed_orders = seed_plan(plan, farm_state, money_proj, slots, day)
    orders.extend(seed_orders)
    slots -= len(seed_orders)
    money_proj -= sum(SEED_COST[o[1]] * o[2] for o in seed_orders)

    if ANIMALS_ENABLED:
        fert_b = []
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

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
P_FEED         = 9250
P_COLLECT      = 9200
P_CARE         = 9150
P_WATER_WINDOW = 8900   # bonus-window watering of one-time crops
P_FERTILIZE    = 8800
P_WATER        = 8600
P_DIG_SPENT    = 7800
P_PLANT        = 7000
P_DIG_WEED     = 6400

_PRICE_HIST = {}        # product -> (day, price) recorded at hour 0


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
        for s in remaining:
            if product in SHOPS[s]:
                rate += (12.0 if len(SHOPS[s]) == 1 else 6.0) * k_future / len(remaining)
    return rate


def drain_between(day, product, days, town):
    total = 0.0
    for d in range(day, min(day + days, LAST_DAY)):
        total += town_rate(d, product, town)
    return total


def project_price(item, inv, day, days, town, opp_rate, params):
    """Expected price `days` from now given town drain and opponent sales."""
    drain = drain_between(day, item, days, town)
    inv_f = max(1.0, inv - drain + opp_rate * days)
    return price_at(item, inv_f, params)


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
                        out["daily_rate"][crop] += rem * 1.4 / days_left
                elif t.get("yield_units", 0) > 0:
                    out["daily_rate"][crop] += max(1.0, t["yield_units"] * 0.7) / days_left
            elif "animal" in t:
                a = t["animal"]
                out["animal_counts"][a] += 1
                p = ANIMAL_PRODUCT[a]
                out["daily_rate"][p] += (1.0 + 1.0 / ANIMAL_INTERVAL[a]) / days_left
                out["daily_rate"]["FERTILIZER"] += 0.8 / days_left
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


def _fert_marginal(crop, day, prices, params):
    """Dollar value of fertilizing one plant of `crop` vs selling the fert."""
    if crop not in ("STRAWBERRY", "TOMATO", "WHEAT"):
        return -1.0
    est_p = prices.get(crop, BASE_PRICE[crop])
    if crop == "STRAWBERRY":
        added = 2
    elif crop == "TOMATO":
        added = 3
    else:
        added = 2 if MAX_YIELD[crop] - units_estimate(crop, day, False) >= 2 else 0
    return added * est_p - prices.get("FERTILIZER", BASE_PRICE["FERTILIZER"])


def wave_profit(crop, day, prices, inv, params, town, opp):
    """Expected profit of one wave of `crop` planted today (per tile)."""
    if not can_finish(crop, day):
        return -1e9
    with_fert = _fert_marginal(crop, day, prices, params) > 0
    units = units_estimate(crop, day, with_fert)
    if units <= 0:
        return -1e9
    horizon = FIRST_YIELD[crop] if ONGOING[crop] else MAX_YIELD_DAY[crop]
    est_p = project_price(crop, inv.get(crop, 10000), day, horizon, town,
                          opp.get("daily_rate", {}).get(crop, 0.0), params)
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


def allowance_tiles(crop, day, prices, inv, params, town, opp):
    """Max tiles of `crop` we may plant without self-crashing the market,
    given remaining town demand and opponent production."""
    tol = glut_tolerance(crop, params)
    drain_rem = drain_between(day, crop, LAST_DAY - day, town)
    opp_est = opp["daily_rate"].get(crop, 0.0) * (LAST_DAY - day)
    allowance = max(30.0, tol + drain_rem - opp_est)
    per_tile = max(1, units_estimate(crop, day, False) * _waves_left(crop, day))
    return max(1.0, allowance / per_tile)


def crop_plan(farm_state, market_state, opp, day, prices, inv, params, money,
              land_extra=0):
    tiles_avail = farm_state["empty"] + farm_state["weeds"] + len(farm_state["spent"]) \
        + land_extra
    if tiles_avail <= 0:
        return {"crops": {}, "ranked": [], "profit": {}, "fert_reserve": 0,
                "tiles_avail": 0}

    # Labor cap: every tile costs ~1 action/day (water/harvest/fert); keep the
    # farm at a size the hired crew can actually care for.  Existing plants
    # count against the cap.
    labor_cap = 30 if day <= 10 else (26 if day <= 20 else 20)
    in_field = sum(farm_state["crop_counts"].values())
    tiles_avail = min(tiles_avail, max(0, labor_cap - in_field))

    # Seasonality: crops whose payoff window has passed are never worth
    # planting this late.
    late_skip = {"MELON": 9, "TOMATO": 20, "STRAWBERRY": 22,
                 "CARROT": 24, "WHEAT": 27}

    profits = {}
    horizon = {}
    for c in CROPS_LIST:
        if day > late_skip.get(c, 30):
            continue
        # Dead market: never add supply to a crashed price, except the feed
        # staple wheat (its floor price is all we ever get anyway).
        if c != "WHEAT" and prices.get(c, BASE_PRICE[c]) < 0.6 * BASE_PRICE[c]:
            continue
        profits[c] = wave_profit(c, day, prices, inv, params, market_state["town"], opp)
        horizon[c] = FIRST_YIELD[c] if ONGOING[c] else MAX_YIELD_DAY[c]
    ranked = [c for c in CROPS_LIST if profits.get(c, 0) > 0]
    ranked.sort(key=lambda c: profits[c] / max(1, horizon[c]), reverse=True)

    plan = {}
    remaining = tiles_avail

    # Early-game cash block: cheap fast crops so money arrives in time for
    # land and animals.  Wheat/carrot pay out from day 2-4 while premium
    # crops sit dormant until day 8-13.
    cheap = [c for c in ("WHEAT", "CARROT") if profits.get(c, 0) > 0]
    if day <= 6 and money < 2500 and cheap:
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
                                  market_state["town"], opp), HARD_CAPS[c])
        take = int(min(cap, remaining))
        if take <= 0:
            continue
        plan[c] = take
        remaining -= take

    fert_reserve = 0
    for c, n in plan.items():
        if c == "STRAWBERRY" and _fert_marginal(c, day, prices, params) > 0:
            fert_reserve += 3 * n
        elif c == "TOMATO" and _fert_marginal(c, day, prices, params) > 0:
            fert_reserve += 1 * n
    for (x, y, t) in farm_state["plants"]:
        c = t["crop"]
        if c == "STRAWBERRY" and _fert_marginal(c, day, prices, params) > 0 \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 2
        elif c == "TOMATO" and _fert_marginal(c, day, prices, params) > 0 \
                and _productions_remaining(c, t["planted_day"], day) > 0:
            fert_reserve += 1

    return {"crops": plan, "ranked": ranked, "profit": profits,
            "fert_reserve": min(fert_reserve, FERT_USE_MAX * 2),
            "tiles_avail": tiles_avail}


# ============================================================================
# Animal economics
# ============================================================================
def animal_value(animal, day, prices, wheat_cost, fert_est):
    days_left = LAST_HARVEST - day - ANIMAL_FIRST[animal]
    if days_left <= 2:
        return -1e9
    p = ANIMAL_PRODUCT[animal]
    prod_per_day = 1.0 + 1.0 / ANIMAL_INTERVAL[animal]     # with daily care
    daily = prices.get(p, BASE_PRICE[p]) * prod_per_day + fert_est - wheat_cost
    return daily * days_left * 0.85 - ANIMAL_COST[animal]


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
        val = animal_value(a, day, prices, wheat_price * 0.9, fert_est)
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
        inv_p = inv.get(p, 10000)

        if forced or cash_need:
            n = q
        else:
            drain = town_rate(day, p, market_state["town"])
            opp_sell = opp["daily_rate"].get(p, 0.0)
            per_turn = max(0.0, drain - opp_sell) / 24.0
            pace = q / max(1.0, days_left * 24.0)      # shed empties by endgame
            if price >= BASE_PRICE[p] or inv_p < 10000:
                n = min(q, pace + per_turn + 0.5)
            elif price >= 0.8 * BASE_PRICE[p]:
                n = min(q, (pace + per_turn) * 0.7)
            else:
                # crashed to the floor: price will not recover, dump it
                n = q
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
        for _ in range(buy):
            if slots <= 0:
                break
            orders.append(["BUY_SEED", c, 1])
            slots -= 1
            budget -= cost
    return orders


def fert_buy_plan(farm_state, market_state, prices, params, money, slots):
    orders = []
    shed_fert = market_state["shed"].get("FERTILIZER", 0)
    need = market_state["fert_reserve"]
    if shed_fert >= need or slots <= 0 or money < 400:
        return orders
    best = 0.0
    for c in ("STRAWBERRY", "TOMATO"):
        m = _fert_marginal(c, 0, prices, params)
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


def hire_plan(farm_state, market_state, day, hour, money, slots, plan_value_per_action):
    orders = []
    if day >= 29 or hour >= HIRE_HOUR_CUT or slots <= 0 or money < 60:
        return orders
    target = HANDS_MAX
    if plan_value_per_action is not None and plan_value_per_action < HAND_VALUE_MIN:
        target = 3
    elif money < 200:
        target = 2
    elif money < 600:
        target = 3
    elif money < 1200:
        target = 4
    elif money < 2500:
        target = 5

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


def land_plan(farm_state, market_state, day, money, slots, marginal_tile_profit):
    orders = []
    n_extra = max(0, len(market_state["unlocked"]) - 1)
    if n_extra >= 3 or day > LAND_LAST_DAY or slots <= 0:
        return orders
    cost = LAND_CASH[n_extra]
    if money < cost or day < LAND_DAYS[n_extra]:
        return orders
    if marginal_tile_profit <= 0:
        return orders
    # Animals beat land: once we own animals waiting to be placed, spend cash
    # on them instead of a third (or later) quadrant.
    pending = sum(market_state["shed"].get(a, 0) for a in ANIMALS_LIST)
    if day >= 12 and (pending > 0 or farm_state["n_animals"] > 0):
        return orders
    gain = 25 * marginal_tile_profit * 0.55      # discount: travel/labor/weeds
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

    fert_worth = {c: _fert_marginal(c, day, prices, market_state["params"])
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
                elif not t.get("cared_today", False):
                    p = ANIMAL_PRODUCT[crop]
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
        score = prio - TRAVEL_COST * _manhattan(ux, uy, x, y) \
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
                        and _fert_marginal(crop, day, prices, params) > 0:
                    return ["FERTILIZE"]
            else:
                if ripe and age >= MAX_YIELD_DAY[crop]:
                    return ["HARVEST"]
                if not tile.get("watered_today", True):
                    return ["WATER"]
                if my_inv.get("FERTILIZER", 0) > 0 \
                        and tile.get("fertilized_until_day", -1) < day \
                        and _fert_marginal(crop, day, prices, params) > 0:
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
            if not tile.get("cared_today", False):
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
        for p in PRODUCTS:
            if p in prices:
                _PRICE_HIST[p] = (day, prices[p])

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
                            slots, plan_value_per_action)
    orders.extend(hire_orders)
    slots -= len(hire_orders)

    # ---- land: buying land expands the crop plan for today -----------------
    profits = [plan["profit"].get(c, 0) for c in plan["crops"]]
    marginal_tile_profit = min(profits) if profits else 0
    land_orders = land_plan(farm_state, market_state, day, money_proj, slots,
                            marginal_tile_profit)
    if land_orders:
        n_extra = max(0, len(market_state["unlocked"]) - 1)
        money_proj -= LAND_CASH[n_extra]
        plan = crop_plan(farm_state, market_state, opp, day, prices, inv,
                         params, money_proj, land_extra=24)
        market_state["fert_reserve"] = plan["fert_reserve"]
        profits = [plan["profit"].get(c, 0) for c in plan["crops"]]
        marginal_tile_profit = min(profits) if profits else 0
    orders.extend(land_orders)
    slots -= len(land_orders)

    seed_orders = seed_plan(plan, farm_state, money_proj, slots, day)
    orders.extend(seed_orders)
    slots -= len(seed_orders)
    money_proj -= sum(SEED_COST[o[1]] * o[2] for o in seed_orders)

    if ANIMALS_ENABLED:
        fert_b = fert_buy_plan(farm_state, market_state, prices, params,
                               money_proj, slots)
        orders.extend(fert_b)
        slots -= len(fert_b)
        money_proj -= len(fert_b) * prices.get("FERTILIZER", 100)
        wheat_b = wheat_buy_plan(farm_state, market_state, prices,
                                 money_proj, slots)
        orders.extend(wheat_b)
        slots -= len(wheat_b)
        money_proj -= len(wheat_b) * prices.get("WHEAT", 25)
        animal_b = animal_plan(farm_state, market_state, opp, day, prices,
                               inv, params, money_proj, slots)
        orders.extend(animal_b)
        slots -= len(animal_b)
        money_proj -= sum(ANIMAL_COST[o[1]] for o in animal_b)

    if slots < 0:
        orders = orders[:MAX_ORDERS]

    # ---- unit scheduling ------------------------------------------------
    tasks, plant_map, build_map = build_tasks(farm, farm_state, plan,
                                              market_state, day, prices)
    fa, hand_actions = schedule_units(farm, farm_state, market_state, plan,
                                      tasks, plant_map, build_map, day, prices)

    return {"farmer": fa, "hands": hand_actions, "market": orders}

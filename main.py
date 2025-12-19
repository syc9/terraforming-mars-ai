import random

class Card:
    def __init__(self, name, cost, effects=None, tags=None, conditions=None, actions=None):
        self.name = name
        self.cost = cost
        self.effects = effects or {}
        self.tags = tags or []
        if isinstance(tags, str):
            self.tags = [tags]
        # condition is a dict with possible keys: "global_params_min", "global_params_max", "tags_played", "production", "resources"
        self.conditions = conditions or {} 
        # action is a list
        self.actions = actions or []
        self.action_used = False # check if action was used per generation

    def can_play(self, player, game_state):
        # Check global parameters MIN
        global_params_min_cond = self.conditions.get("global_params_min", {})
        for param, min_val in global_params_min_cond.items():
            if game_state.get(param, None) is None or game_state[param] < min_val:
                return False
        
        # Check global parameters MAX
        global_params_max_cond = self.conditions.get("global_params_max", {})
        for param, max_val in global_params_max_cond.items():
            if game_state.get(param, None) is None or game_state[param] > max_val:
                return False

        # Check tags played
        tags_played_cond = self.conditions.get("tags_played", {})
        for tag, min_count in tags_played_cond.items():
            count = sum(1 for card in player.played_cards if tag in card.tags)
            if count < min_count:
                return False
        
        # Check resource production
        production_cond = self.conditions.get("production", {})
        for res, min_prod in production_cond.items():
            if player.production.get(res, 0) < min_prod:
                return False
            
        # Check resources available
        resource_cond = self.conditions.get("resources", {})
        for res, min_res in resource_cond.items():
            if player.resources.get(res, 0) < min_res:
                return False
        
        # All conditions met
        return True

    def apply_effects(self, player, game_state, effects_override=None):
        effects_queue = effects_override.items() if effects_override else self.effects.items()

        for effect, value in effects_queue:
            if effect.endswith("_production"):
                res = effect.replace("_production", "")
                player.production[res] += value

            elif effect.startswith("gain_"):
                res = effect.replace("gain_", "")
                player.resources[res] += value

            elif effect.startswith("resource_value_"):
                res = effect.replace("resource_value_", "")
                player.resource_value[res] += value

            elif effect == "increase_TR":
                player.terraform_rating += value
            
            elif effect == "raise_oxygen":
                for _ in range(value):
                    old_oxygen = game_state["oxygen"]
                    game_state["oxygen"] = min(14, game_state["oxygen"] + value)
                    net_increase = game_state["oxygen"] - old_oxygen
                    player.terraform_rating += net_increase

                    # Trigger raise_temperature if oxygen crosses to 8
                    if old_oxygen < 8 <= game_state["oxygen"]:
                        print("Oxygen reached 8 — triggering raise_temperature effect")
                        self.apply_effects(player, game_state, effects_override={"raise_temperature": 1})

            elif effect == "raise_temperature":
                for _ in range(value):
                    if game_state["temperature"] < 8:
                        old_temp = game_state["temperature"]
                        game_state["temperature"] = min(8, game_state["temperature"] + 2)

                        # Heat production milestones
                        if old_temp < -24 <= game_state["temperature"]:
                            player.production["heat"] += 1
                        if old_temp < -20 <= game_state["temperature"]:
                            player.production["heat"] += 1

                        # Trigger place_ocean at 0°C
                        if old_temp < 0 <= game_state["temperature"]:
                            print("Temperature reached 0°C — triggering place_ocean")
                            self.apply_effects(player, game_state, effects_override={"place_ocean": 1})

                        net_increase = (game_state["temperature"] - old_temp) // 2
                        player.terraform_rating += net_increase

            elif effect == "place_ocean":
                original_oceans = game_state["oceans"]
                game_state["oceans"] = min(9, game_state["oceans"] + value)
                net_increase = game_state["oceans"] - original_oceans
                player.terraform_rating += net_increase

                # If placing ocean causes another effect, trigger it here
                # e.g., draw a card if card effects specify it in future


class Player:
    def __init__(self, name):
        self.name = name
        self.resources = {
            "megacredits": 40,
            "steel": 0,
            "titanium": 0,
            "plants": 0,
            "energy": 0,
            "heat": 0
        }
        self.production = {
            "megacredits": 0,
            "steel": 0,
            "titanium": 0,
            "plants": 0,
            "energy": 0,
            "heat": 0
        }
        self.resource_value = {
            "megacredits": 0,
            "steel": 2,
            "titanium": 3,
            "plants": 0,
            "energy": 0,
            "heat": 0
        }
        self.played_tags = {
            "building": 0,       # steel can be used to pay for 2 megacredits
            "space": 0,          # titanium can be used to pay as 3 megacredits
            "science": 0,        # science tags
            "power": 0,          # power tags
            "earth": 0,          # earth tags
            "jovian": 0,         # jovian tags
            "plant": 0,          # plant tags
            "microbe": 0,        # microbe tags
            "city": 2,           # city tags
            "event": 0,          # event tags
            "Standard": 0        # standard project 
        }

        self.game_effects = {
            "Standard Technology": False,   # gain 3 MC after a standard project is played
            "Earth Catapult": False    # When you play a card, you pay 2 MC less for it
        }

        # for ML training purposes
        self.weights = {
            "temperature": 15,  # global params
            "oxygen": 15,
            "oceans": 15,
            "megacredits": 7,   # resources + production
            "steel": 5,
            "titanium": 5,
            "plants": 7,
            "energy": 7,
            "heat": 7,
            "building": 2,       # tags
            "space": 2,          
            "science": 3,        
            "power": 2,          
            "earth": 1,          
            "jovian": 1,         
            "plant": 2,          
            "microbe": 1,        
            "city": 1,           
            "event": 1,          
            "Standard": 1         
        }

        self.hand = []
        self.played_cards = []

        self.actions = []
        self.played_actions = []
        self.terraform_rating = 14  # Solo mode starts at 14

    def produce_resources(self):

        # Resources: standard production (if megacredits, add TR as well)
        for res, prod in self.production.items():
                self.resources[res] += prod + (res=="megacredits")*self.terraform_rating

        # Convert all energy to heat at the end of production
        self.resources["heat"] += self.resources["energy"]
        self.resources["energy"] = 0

    def resource_discount(self, tag, card):
        total_cost = card.cost
        res_discount = 0

        # apply resources if has specific tag
        tag_pairing = {"space": "titanium", "building": "steel"}
        res = tag_pairing.get(tag)

        res_value = self.resource_value[res]
        res_max = self.resources[res]

        # how many resources needed to cover the total cost (round up)
        res_needed = (total_cost + res_value - 1) // res_value

        res_used = min(res_max, res_needed)
        res_discount = res_used * res_value
        # [Note: not included in return] reduce cost to 0 (even if overpays)
        # total_cost = max(0, total_cost - res_discount)

        # return the amount of resources used and the total value of the discount
        return [res_used, res_discount]

    def play_card(self, card, played_actions, game_state):
        total_cost = card.cost

        # Apply titanium if card has "space" tag
        if "space" in card.tags and self.resources["titanium"] > 0:
            [titanium_used, titanium_discount] = self.resource_discount("space", card)
            total_cost = max(0, total_cost - titanium_discount)
            self.resources["titanium"] -= titanium_used

            played_actions.append(f"Used {titanium_used} titanium to reduce cost by {titanium_discount} MC (possibly overpaying).")

        # Apply steel if card has "building" tag
        if "building" in card.tags and self.resources["steel"] > 0:
            [steel_used, steel_discount] = self.resource_discount("building", card)
            total_cost = max(0, total_cost - steel_discount)
            self.resources["steel"] -= steel_used

            played_actions.append(f"Used {steel_used} steel to reduce cost by {steel_discount} MC (possibly overpaying).")

        if self.resources["megacredits"] < total_cost:
            played_actions.append(f"Not enough megacredits to play {card.name}. Needed: {total_cost}, Available: {self.resources['megacredits']}")
            return
        
        self.resources["megacredits"] -= total_cost
        self.played_cards.append(card)

        for tag in card.tags:
            self.played_tags[tag] += 1

        # trigger global effect if card is played
        if card.name in self.game_effects:
            self.game_effects[card.name] = True

        self.actions.append(card.actions)
        card.apply_effects(self, game_state)

        if "Standard" in card.tags:
            if self.game_effects["Standard Technology"]:
                self.resources["megacredits"] += 3
                played_actions.append(f"{self.name} gains 3 MC due to Standard Technology triggered by playing {card.name}.")

        # always should have all the standard project cards available to play
        if "Standard" not in card.tags:
            self.hand.remove(card)        
        return


    def evaluate_card(self, card, game_state, generation):
        if not card.can_play(self, game_state):
            return -1  # Can't play due to conditions

        net_cost = card.cost
        for tag in ["space", "building"]:
            if tag in card.tags:
                net_cost = max(0, net_cost - self.resource_discount(tag, card)[1])
        if net_cost > self.resources["megacredits"]:
            return -1  # Can't afford it

        max_generations = 14
        progress = generation / max_generations
        remaining = max_generations - generation

        # Production weight decreasing over time
        phase_prod_factor = 1 - 0.9* progress

        # Lump resources become more valuable later
        phase_resource_factor = 0.6 + 1.2*progress

        # Synergy weight
        synergy_weight = max(0.05, 1 - progress**2)

        def param_completion(name, val, max_val, min_val=0):
            return (game_state[name] - min_val) / (max_val - min_val)
        
        oxygen_pct = param_completion("oxygen", game_state["oxygen"], 14)
        temp_pct = param_completion ("temperature", game_state["temperature"], 8, -30)
        ocean_pct = param_completion("oceans", game_state["oceans"], 9)

        def global_param_weight(pct):
            return 1.5 * (1 - pct) ** 1.25

        gp_score = 0
        prod_score = 0
        gen_score = 0
        tag_score = 0
        action_score = 0

        # effects scoring
        for effect, val in card.effects.items():
            # Global parameter effects
            if effect == "raise_oxygen" and game_state["oxygen"] < 14:
                gp_score += val * self.weights.get("oxygen") * global_param_weight(oxygen_pct) * generation
            if effect == "raise_temperature" and game_state["temperature"] < 8:
                gp_score += val * self.weights.get("temperature") * global_param_weight(temp_pct) * generation
            if effect == "place_ocean" and game_state["oceans"] < 9:
                gp_score += val * self.weights.get("oceans") * global_param_weight(ocean_pct) * generation

            # Resource production effects
            if effect.endswith("_production"):
                res = effect.replace("_production", "")
                base_factor = self.weights.get(res) * 0.1
                prod_score += val * base_factor * remaining * phase_prod_factor

            # Immediate gain effects
            if effect.startswith("gain_"):
                res = effect.replace("gain_", "")
                val_weight = self.weights.get(res) * 0.1
                gen_score += val * val_weight * phase_resource_factor

            for t in card.tags:
                tag_score += self.weights.get(t) * synergy_weight
            

        if "Standard" in card.tags and self.game_effects["Standard Technology"]: 
            tag_score += (0.7*generation)*3
        
        # Evaluate each action on the card
        for action in card.actions:
            effect = action.get("effect", {})
            cost = action.get("cost", {})

            effect_name, effect_val = list(effect.items())[0]
            value_score = 0

            if effect_name == "place_ocean" and game_state["oceans"] < 9:
                value_score += effect_val * self.weights.get("oxygen") * global_param_weight(oxygen_pct) * remaining
            if effect_name == "raise_oxygen" and game_state["oxygen"] < 14:
                value_score += effect_val * self.weights.get("temperature") * global_param_weight(temp_pct) * remaining
            if effect_name == "raise_temperature" and game_state["temperature"] < 8:
                value_score += effect_val * self.weights.get("oceans") * global_param_weight(ocean_pct) * remaining
            if effect_name.endswith("_production"):
                res = effect_name.replace("_production", "")
                base_factor = self.weights.get(res) * 0.1
                value_score += val * base_factor * remaining * phase_prod_factor
            if effect_name.startswith("gain_"):
                res = effect_name.replace("gain_", "")
                val_weight = self.weights.get(res) * 0.2
                value_score += val * val_weight * phase_resource_factor

            # Compute cost in MC equivalent
            cost_value = sum(
                amt * (self.resource_value.get(res, 1)) for res, amt in cost.items()
            )

            net_action_value = value_score - cost_value
            # Assume each action can be used once per generation until game ends
            action_score += max(0, net_action_value) * max(1, remaining)

        raw_score = gp_score + prod_score + gen_score + tag_score + action_score
        weighted_score = raw_score / max(1, net_cost)
        return weighted_score

    def take_turn(self, generation, game_state):
        played_actions = []

        while True:
            best_card = None
            best_score = -1

            # Convert heat to temperature if possible
            if self.resources["heat"] >= 8 and game_state["temperature"] < 8:
                new_temp = Card("Increase Temperature", 0, effects={"raise_temperature": 1, "gain_heat": -8})
                self.hand.append(new_temp)
                self.play_card(new_temp, played_actions, game_state)
                played_actions.append(f"{self.name} increased temperature to {game_state['temperature']}")
                continue

            # Convert plants to greenery if possible
            if self.resources["plants"] >= 8:
                new_green = Card("Greenery", 0, effects={"raise_oxygen": 1, "gain_plants": -8})
                self.hand.append(new_green)
                self.play_card(new_green, played_actions, game_state)
                played_actions.append(f"{self.name} placed greenery and raised oxygen to {game_state['oxygen']}")
                continue

            # Evaluate all cards in hand using new scoring model
            for card in self.hand:
                score = self.evaluate_card(card, game_state, generation)
                if score > best_score:
                    best_score = score
                    best_card = card

            # Play best card if found
            if best_card:
                self.play_card(best_card, played_actions, game_state)
                played_actions.append(f"{self.name} played {best_card.name} (weighted score={round(best_score, 2)})")
            else:
                played_actions.append(f"{self.name} passed with {self.resources['megacredits']} MC remaining")
                break  # No good cards or resources left

        return played_actions



class TerraformingMarsSoloGame:
    def __init__(self):
        self.player = Player("Solo Player")
        self.generation = 1
        self.max_generations = 14
        self.global_parameters = {
            "oxygen": 0,         # Target: 14
            "temperature": -30,  # Target: +8
            "oceans": 0          # Target: 9
        }

        self.deck = self.generate_deck(self.player.played_tags)
        self.deal_initial_hand()

    def generate_deck(self, played_tags):
        return [
            Card("Acquired Company", 10, effects={"megacredits_production": 3}, tags=["earth"]),
            # Card("Adaption Technology", 12, effects={"special": 0}, tags=["science"]),
            Card("Adapted Lichen", 9, effects={"plants_production": 1}, tags=["plant"]),
            Card("Advanced Alloys", 9, effects={"resource_value_steel": 1, "resource_value_titanium": 1}, tags=["science"]),
            Card("Aerobraked Amonia Asteroid", 26, effects={"heat_production": 3, "plants_production": 1}, tags=["event", "space"]),
            Card("Algae", 10, effects={"plants_production": 2, "gain_plants": 1}, tags=["plant"], conditions={"global_params_min": {"oceans": 5}}),
            # Card("Aquifer Pumping", 18, tags=["building"], actions=[]),
            Card("Archaebacteria", 6, effects={"plants_production": 1}, tags=["microbe"], conditions={"global_params_max": {"temperature": -18}}),
            # Card("Arctic Algae", 12, effects={"special": 1, "gain_plants": 1}, tags=["plant"], conditions={"global_params_max": {"temperature": -12}}),
            Card("Artificial Lake", 15, effects={"place_ocean": 1}, tags=["building"], conditions={"global_params_min": {"temperature": -6}}),
            Card("Artificial Photosynthesis", 12, effects={"plants_production": 1, "energy_production": 2}, tags=["science"]),
            Card("Asteroid", 14, effects={"raise_temperature": 1, "gain_titanium": 2}, tags=["event", "space"]),
            Card("Asteroid Mining", 30, effects={"titanium_production": 2}, tags=["jovian", "space"]),
            Card("Asteroid Mining Consortium", 13, effects={"titanium_production": 1}, tags=["jovian"], conditions={"production": {"titanium": 1}}),
            Card("Beam from a Thorium Asteroid", 32, effects={"energy_production": 3, "heat_production": 3}, tags=["jovian", "space", "power"], conditions={"tags_played": {"jovian": 1}}),
            Card("Big Asteroid", 27, effects={"raise_temperature": 2, "gain_titanium": 4}, tags=["event", "space"]),
            Card("Biomass Combustors", 4, effects={"energy_production": 2}, tags=["power", "building"], conditions={"global_params_min": {"oxygen": 6}}),
            Card("Black Polar Dust", 15, effects={"place_ocean": 1, "megacredits_production": -2, "heat_production": 3}),
            Card("Bribed Committee", 7, effects={"increase_TR": 2}, tags=["event", "earth"]),
            Card("Building Industries", 6, effects={"energy_production": -1, "steel_production": 2}, tags=["building"], conditions={"production": {"energy": 1}}),
            Card("Bushes", 10, effects={"plants_production": 2, "gain_plants": 2}, tags=["plant"], conditions={"global_params_min": {"temperature": -10}}),
            Card("Callisto Penal Mines", 24, effects={"megacredits_production": 3}, tags=["space", "jovian"]),
            Card("Capital", 26, effects={"energy_production": -2, "megacredits_production": 5, "gain_megacredits": 3}, tags=["city", "building"], conditions={"global_params_min": {"oceans": 4}, "production": {"energy": 2}}),
            Card("Carbonate Processing", 6, effects={"energy_production": -1, "heat_production": 3}, tags=["building"], conditions={"production": {"energy": 1}}),
            Card("Cartel", 8, effects={"megacredits_production": played_tags["earth"]}, tags=["earth"]),
            Card("Cloud Seeding", 11, effects={"megacredits_production": -1, "plants_production": 2}, conditions={"global_params_min": {"oceans": 3}}),
            Card("Comet", 21, effects={"raise_temperature": 1, "place_ocean": 1}, tags=["event", "space"]),
            Card("Commerical District", 16, effects={"energy_production": -1, "megacredits_production": 4, "gain_megacredits": 3}, tags=["building", "city"], conditions={"production": {"energy": 1}}),
            Card("Convoy from Europa", 15, effects={"place_ocean": 1}, tags=["event", "space"]),
            Card("Cupola City", 16, effects={"energy_production": -1, "megacredits_production": 3, "gain_megacredits": 3}, tags=["building", "city"], conditions={"global_params_max": {"oxygen": 9}, "production": {"energy": 1}}),
            Card("Deep Well Heating", 13, effects={"energy_production": 1, "raise_temperature": 1}, tags=["building", "power"]),
            Card("Deimos Down", 31, effects={"raise_temperature": 3, "gain_steel": 4}, tags=["event", "space"]),
            Card("Designed Microorganisms", 16, effects={"plants_production": 2}, tags=["science", "microbe"], conditions={"global_params_max": {"temperature": -14}}),
            Card("Domed Crater", 24, effects={"energy_production": -1, "megacredits_production": 3, "gain_megacredits": 3, "gain_plants": 3}, tags=["building", "city"], conditions={"global_params_max": {"oxygen": 7}, "production": {"energy": 1}}),
            # Card("Earth Catapult", 23, effects={"special": 2}, tags=["earth"]),
            # Card("Earth Office", 1, effects={"special": 3}, tags=["earth"]),
            Card("Energy Saving", 15, effects={"energy_production": played_tags["city"]}, tags=["power"]),
            Card("Energy Tapping", 3, effects={"energy_production": 1}, tags=["power"]),
            Card("Eos Chasma National Park", 16, effects={"gain_plants": 3, "megacredits_production": 2}, tags=["building", "plant"], conditions={"global_params_min": {"temperature": -12}}),
            # Card("Extreme-Cold Fungus", 13, tags=["microbe"], actions=[]),
            Card("Farming", 16, effects={"megacredits_production": 2, "plants_production": 2, "gain_plants": 2}, tags=["plant"], conditions={"global_params_min": {"temperature": 4}}),
            Card("Flooding", 7, effects={"place_ocean": 1}, tags=["event"]),
            Card("Food Factory", 12, effects={"plants_production": -1, "megacredits_production": 4}, tags=["building"], conditions={"production": {"plants": 1}}),
            Card("Fuel Factory", 6, effects={"energy_production": -1, "titanium_production": 1, "megacredits_production": 1}, tags=["building"], conditions={"production": {"energy": 1}}),
            Card("Fueled Generators", 1, effects={"megacredits_production": -1, "energy_production": 1}, tags=["building", "power"]),
            Card("Fusion Power", 14, effects={"energy_production": 3}, tags=["science", "building", "power"], conditions={"tags_played": {"power": 2}}),
            Card("Gene Repair", 12, effects={"megacredits_production": 2}, tags=["science"], conditions={"tags_played": {"science": 3}}),
            Card("Geothermal Power", 11, effects={"energy_production": 2}, tags=["building", "power"]),
            Card("GHG Factories", 11, effects={"energy_production": -1, "heat_production": 4}, tags=["building"], conditions={"production": {"energy": 1}}),
            # Card("GHG Producing Bacteria", 8, tags=["science", "microbe"], conditions={"global_params_min": {"oxygen": 4}}, actions=[]),
            Card("Giant Ice Asteroid", 36, effects={"raise_temperature": 2, "place_ocean": 2}, tags=["event", "space"]),
            Card("Giant Space Mirror", 17, effects={"energy_production": 3}, tags=["space", "power"]),
            Card("Grass", 11, effects={"plants_production": 1, "gain_plants": 3}, tags=["plant"], conditions={"global_params_min": {"temperature": -16}}),
            Card("Giant Escarpment Consortium", 6, effects={"steel_production": 1}, conditions={"production": {"steel": 1}}),
            Card("Greenhouses", 6, effects={"gain_plants": played_tags["city"]}, tags=["building", "plant"]),
            Card("Heather", 6, effects={"gain_plants": 1, "plants_production": 1}, tags=["plant"], conditions={"global_params_min": {"temperature": -14}}),
            Card("Hired Raiders", 1, effects={"gain_steel": 2, "gain_megacredits": 3}, tags=["event"]),
            Card("Ice Asteroid", 23, effects={"place_ocean": 2}, tags=["event", "space"]),
            Card("Ice Cap Melting", 5, effects={"place_ocean": 1}, tags=["event"], conditions={"global_params_min": {"temperature": 2}}),
            # Card("Immigrant City", effects={"special": 4}, tags=["city", "building"], conditions={"production": {"energy": 1}}),
            Card("Immigration Shuttles", 31, effects={"megacredits_production": 5}, tags=["building", "space"]),
            Card("Import of Advanced GHG", 9, effects={"heat_production": 2}, tags=["event", "space", "earth"]),
            Card("Imported GHG", 7, effects={"heat_production": 1, "gain_heat": 3}, tags=["event", "space", "earth"]),
            Card("Imported Hydrogen", 16, effects={"place_ocean": 1, "gain_plants": 3}, tags=["event", "space", "earth"]),
            Card("Imported Nitrogen", 23, effects={"increase_TR": 1, "gain_plants": 4}, tags=["event", "space", "earth"]),
            Card("Industrial Center", 4, effects={"gain_megacredits": 3, "steel_production": 1}, tags=["building"]),
            Card("Industrial Microbes", 12, effects={"energy_production": 1, "steel_production": 1}, tags=["building", "microbe"]),
            Card("Insects", 9, effects={"plants_production": played_tags["plant"]}, tags=["microbe"], conditions={"global_params_min": {"oxygen": 6}}),
            Card("IO Mining Industries", 41, effects={"titanium_production": 2, "megacredits_production": 2}, tags=["jovian", "space"]),
            Card("Ironworks", 11, effects={"raise_oxygen": 1, "gain_steel": 1, "gain_energy": -4}, tags=["building"], conditions={"resources": {"energy": 4}}),
            Card("Kelp Farming", 17, effects={"plants_production": 3, "megacredits_production": 2, "gain_plants": 2}, tags=["plant"], conditions={"global_params_min": {"oceans": 6}}),
            Card("Lake Mariners", 18, effects={"place_ocean": 2}, conditions={"global_params_min": {"temperature": 0}}),
            Card("Large Convoy", 36, effects={"place_ocean": 1, "gain_plants": 5}, tags=["event", "space", "earth"]),
            Card("Lava Flows", 18, effects={"raise_temperature": 2, "gain_megacredits": 2}, tags=["event"]),
            Card("Lichen", 7, effects={"plants_production": 1}, tags=["plant"], conditions={"global_params_min": {"temperature": -24}}),
            Card("Lightning Harvest", 8, effects={"megacredits_production": 1, "energy_production": 1}, tags=["power"], conditions={"tags_played": {"science": 3}}),
            Card("Lunar Beam", 13, effects={"megacredits_production": -2, "heat_production": 2, "energy_production": 2}, tags=["earth", "power"]),
            Card("Magnetic Field Dome", 5, effects={"energy_production": -2, "plants_production": 1}, tags=["building"], conditions={"production": {"energy": 2}}),
            Card("Magnetic Field Generators", 20, effects={"energy_production": -4, "plants_production": 2, "increase_TR": 3}, tags=["building"], conditions={"production": {"energy": 4}}),
            Card("Mangrove", 12, effects={"raise_oxygen": 1, "gain_plants": 2}, tags=["plant"], conditions={"global_params_min": {"temperature": 4}}),
            Card("Mass Convertor", 8, effects={"energy_production": 6}, tags=["science", "power"], conditions={"tags_played": {"science": 5}}),
            Card("Media Archives", 8, effects={"gain_megacredits": played_tags["event"]}, tags=["earth"]),
            # Card("Media Group", 6, effects={"special": 5}, tags=["earth"]),
            Card("Medical Lab", 13, effects={"megacredits_production": int(played_tags["building"])}, tags=["science", "building"]),
            Card("Methane from Titan", 28, effects={"heat_production": 2, "plants_production": 2}, tags=["space", "jovian"], conditions={"global_params_min": {"oxygen": 2}}),
            Card("Micro-mills", 3, effects={"heat_production": 1}),
            Card("Mine", 4, effects={"steel_production": 1}, tags=["building"]),
            Card("Mineral Deposit", 5, effects={"gain_steel": 5}, tags=["event"]),
            Card("Mining Exposition", 12, effects={"raise_oxygen": 1, "gain_steel": 2}, tags=["event"]),
            Card("Moss", 4, effects={"plants_production": 1, "gain_plants": -1}, tags=["plant"], conditions={"global_params_min": {"oceans": 3}, "resources": {"plants": 1}}),
            Card("Natural Preserve", 9, effects={"gain_plants": 2, "megacredits_production": 1}, tags=["building", "science"], conditions={"global_params_max": {"oxygen": 4}}),
            Card("Nitrogen-Rich Asteroid", 31, effects={"increase_TR": 2, "raise_temperature": 1, "plants_production": 3}, tags=["event", "space"]),
            Card("Nitrophilic Moss", 8, effects={"plants_production": 2, "gain_plants": -2}, tags=["plant"], conditions={"global_params_min": {"oceans": 3}, "resources": {"plants": 2}}),
            Card("Noctis City", 18, effects={"energy_production": 1, "megacredits_production": 3, "gain_megacredits": 3}, tags=["city", "building"], conditions={"production": {"energy": 1}}),
            Card("Nuclear Farming", 10, effects={"megacredits_production": -2, "energy_production": 3}, tags=["building", "power"]),
            Card("Nuclear Zone", 10, effects={"raise_temperature": 2, "gain_plants": 2}, tags=["earth"]),
            Card("Open City", 23, effects={"gain_plants": 2, "gain_megacredits": 3, "energy_production": -1, "megacredits_production": 4}, tags=["city", "building"], conditions={"global_params_min": {"oxygen": 12}, "production": {"energy": 1}}),
            # Card("Optimal Aerobraking", 7, effects={"gain_megacredits": 3, "gain_heat": 3}, tags=["space"]),
            Card("Ore Processor", 13, effects={"raise_oxygen": 1, "gain_titanium": 1, "gain_energy": -4}, tags=["building"], conditions={"resources": {"energy": 4}}),
            Card("Permafrost Extraction", 8, effects={"place_ocean": 1}, tags=["event"], conditions={"global_params_min": {"temperature": -8}}),
            Card("Peroxide Power", 7, effects={"megacredits_production": -1, "energy_production": 2}, tags=["building", "power"]),
            Card("Phobos Space Haven", 25, effects={"titanium_production": 1, "gain_megacredits": 3}, tags=["city", "space"]),
            Card("Plantation", 15, effects={"raise_oxygen": 1, "gain_plants": 1}, tags=["plant"], conditions={"tags_played": {"science": 2}}),
            Card("Power Grid", 18, effects={"energy_production": played_tags["power"]}, tags=["power"]),
            Card("Power Plant", 4, effects={"energy_production": 1}, tags=["building", "power"]),
            Card("Power Supply Consortium", 5, effects={"energy_production": 1}, tags=["power"], conditions={"tags_played": {"power": 2}}),
            Card("Protected Valley", 23, effects={"raise_oxygen": 1, "megacredits_production": 2, "gain_plants": 2}, tags=["building", "plant"]),
            Card("Quantum Extractor", 13, effects={"energy_production": 4}, tags=["science", "power"], conditions={"tags_played": {"science": 4}}),
            Card("Rad-Chem Factory", 8, effects={"energy_production": -1, "increase_TR": 2}, tags=["building"], conditions={"production": {"energy": 1}}),
            Card("Rad-Suits", 6, effects={"megacredits_production": 1}),
            Card("Release of Inert Gases", 14, effects={"increase_TR": 2}, tags=["event"]),
            # Card("Research Outpost", 18, effects={"special": 6}, tags=["city", "science", "building"]),
            # Card("Robotic Workforce", 9, effects={"special": 7}, tags=["science"]},
            # Card("Rover Construction": 8, effects={"special": 8}, tags=["buliding"]),
            Card("Satellites", 10, effects={"megacredits_production": played_tags["space"]}, tags=["space"]),
            Card("Soil Factory", 9, effects={"energy_production": -1, "plants_production": 1}, tags=["building"], conditions={"production": {"energy": 1}}),
            Card("Solar Power", 11, effects={"energy_production": 1}, tags=["building", "power"]),
            Card("Solar Wind Power", 11, effects={"energy_production": 1, "gain_titanium": 2}, tags=["science", "space", "power"]),
            Card("Soletta", 35, effects={"heat_production": 7}, tags=["space"]),
            # Card("Space Elevator", 27, effects={"titanium_production": 1}, tags=["space", "building"], actions=[]),
            # Card("Space Mirrors", 3, tags=["power", "space"], actions=[]),
            # Card("Space Station", 10, effects={"special": 9}, tags=["space"]),
            Card("Sponsors", 6, effects={"megacredits_production": 2}, tags=["earth"]),
            Card("Steelworks", 15, effects={"raise_oxygen": 1, "gain_steel": 2, "gain_energy": -4}, tags=["building"], conditions={"resources": {"energy": 4}}),
            Card("Strip Mine", 25, effects={"steel_production": 2, "titanium_production": 1, "raise_oxygen": 2, "energy_production": -2}, tags=["building"], conditions={"production": {"energy": 2}}),
            Card("Subterranean Reservoir", 11, effects={"place_ocean": 1}, tags=["event"]),
            Card("Tectonic Stress Power", 18, effects={"energy_production": 3}, tags=["building", "power"], conditions={"tags_played": {"science": 2}}),
            Card("Terraforming Ganymede", 33, effects={"increase_TR": played_tags["jovian"]}, tags=["space", "jovian"]),
            Card("Titanium Mine", 7, effects={"titanium_production": 1}, tags=["building"]),
            Card("Towing a Comet", 23, effects={"raise_oxygen": 1, "place_ocean": 1, "gain_plants": 2}, tags=["event", "space"]),
            Card("Trans-Neptune Probe", 6, tags=["space", "science"]),
            Card("Trees", 13, effects={"plants_production": 3, "gain_plants": 1}, conditions={"global_params_min": {"temperature": -4}}),
            Card("Tropical Resort", 13, effects={"heat_production": -2, "megacredits_production": 3}, tags=["building"], conditions={"production": {"heat": 2}}),
            Card("Tundra Farming", 16, effects={"plants_production": 1, "gain_plants": 1, "megacredits_production": 2}, tags=["plant"], conditions={"global_params_min": {"temperature": -6}}),
            Card("Urbanized Area", 10, effects={"energy_production": -1, "megacredits_production": 2, "gain_megacredits": 3}, tags=["city", "building"], conditions={"production": {"energy": 1}}),
            Card("Underground City", 18, effects={"energy_production": -2, "steel_production": 2, "gain_megacredits": 3}, tags=["city", "building"], conditions={"production": {"energy": 2}}),
            # Card("Underground Detonations", 6, tags=["building"], actions=[]),
            Card("Vesta Shipyard", 15, effects={"titanium_production": 1}, tags=["jovian", "space"]), 
            # Card("Viral Enhancers", 9, effects={"special": 10}, tags=["science", "microbe"]),
            # Card("Water Import from Europa", 25, tags=["jovian", "space"], actions=[]),
            Card("Water Splitting Plant", 12, effects={"raise_oxygen": 1, "gain_energy": -3}, tags=["building"], conditions={"global_params_min": {"oceans": 2}, "resources": {"energy": 3}}),
            Card("Wave Power", 8, effects={"energy_production": 1}, tags=["power"], conditions={"global_params_min": {"oceans": 3}}),
            Card("Windmills", 6, effects={"energy_production": 1}, tags=["power", "building"], conditions={"global_params_min": {"oxygen": 7}}),
            Card("Worms", 8, effects={"plants_production": played_tags["microbe"]}, tags=["microbe"], conditions={"global_params_min": {"oxygen": 4}}),
            Card("Zeppelins", 13, effects={"megacredits_production": played_tags["city"]}, conditions={"global_params_min": {"oxygen": 5}}),

            Card("Mining Rights", 4, effects={"steel_production": 1}, tags=["building"]),
            Card("Mining Area", 9, effects={"titanium_production": 1}, tags=["building"]),
            Card("Investment Loan", 3, effects={"gain_megacredits": 10, "megacredits_production": -1}, tags=["event"]),
            Card("Standard Technology", 6, tags=["science"]),
            Card("Mohole Area", 20, effects={"heat_production": 4, "gain_plants": 2}, tags=["building"]),
            Card("Noctis Farming", 10, effects={"megacredits_production": 1, "gain_plants": 2}, tags=["plant", "building"], conditions={"global_params_min": {"temperature": -20}}),
        ] * 1  # Expanded deck

    def StandardProjectDeck(self):
        return [
            Card("Power Plant (s)", 11, effects={"energy_proudction": 1}, tags=["Standard"]),
            Card("Asteroid (s)", 14, effects={"raise_temperature": 1}, tags=["Standard"]),
            Card("Aquifer Pumping (s)", 18, effects={"place_ocean": 1}, tags=["Standard"]),
            Card("Greenery (s)", 23, effects={"raise_oxygen": 1}, tags=["Standard"]),
            # Card("City (s)", 25, effects={"megacredits_production": 1, "gain_megacredits": 3}, tags=["Standard", "city"])
        ]
    
    def deal_initial_hand(self):
        self.player.hand = self.StandardProjectDeck() + random.sample(self.deck, 10)

    def deal_next_hand(self):
        # add a card to the deck
        self.player.hand = self.player.hand + random.sample(self.deck, 1)

    def check_victory(self):
        gp = self.global_parameters
        return gp["oxygen"] >= 14 and gp["temperature"] >= 8 and gp["oceans"] >= 9

    def play_game(self):
        print("Starting Solo Terraforming Mars...\n")
        while self.generation <= self.max_generations:
            print(f"=== Generation {self.generation} ===")

            if self.generation > 1:
                self.deal_next_hand()

            # reset actions on played cards
            for card in self.player.played_cards:
                card.action_used = False

            # Print resource snapshot before actions
            print(f"TR: {self.player.terraform_rating} | Resources before actions: {self.player.resources}")
            print(f"Production before actions: {self.player.production} ")

            for card in self.player.hand:
                print(card.name)

            actions = self.player.take_turn(self.generation, self.global_parameters)
            # Print all actions for the turn
            for action in actions:
                print(action)

            if self.check_victory():
                print(f"\n Victory! All global parameters achieved in Generation {self.generation}.")
                self.end_game()
                return

            self.player.produce_resources()
            
            self.generation += 1
            print(f"\n Global Parameters: Oxygen = {self.global_parameters['oxygen']}/14, Temp = {self.global_parameters['temperature']}°C, Oceans = {self.global_parameters['oceans']}/9")

            print()

        print("Game Over. You ran out of generations before terraforming completed.")
        self.end_game()

    def end_game(self):
        gp = self.global_parameters
        print(f"\n Final Global Parameters: Oxygen = {gp['oxygen']}/14, Temp = {gp['temperature']}°C, Oceans = {gp['oceans']}/9")
        print(f"TR: {self.player.terraform_rating}, Cards Played: {len(self.player.played_cards)}")
        print(f"\n Total tags played: {self.player.played_tags}")


if __name__ == "__main__":
    game = TerraformingMarsSoloGame()
    game.play_game()

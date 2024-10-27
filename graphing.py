pieces_str_lit = 'total_pieces'
price_str_lit = 'total_price'

def analyze_age_value(lego_sets) -> dict[int, dict[str, int]]:
    
    age_groups = {}
    
    for lego_set in lego_sets:
        age = lego_set.get_age_group()
        pieces = lego_set.get_pieces()
        price = lego_set.get_price()
        
        if age not in age_groups:
            age_groups[age] = {pieces_str_lit: 0, price_str_lit: 0}
            
        age_groups[age][pieces_str_lit] += pieces
        age_groups[age][price_str_lit] += price
    
    best_ratio = 0
    best_age = None
    
    for age, data in age_groups.items():
        if data[price_str_lit] > 0:  # 0 div exeption
            pieces_per_dollar = data[pieces_str_lit] / data[price_str_lit]
            if pieces_per_dollar > best_ratio:
                best_ratio = pieces_per_dollar
                best_age = age
    
    # debug
    for age, data in age_groups.items():
        if data[price_str_lit] > 0:
            ratio = data[pieces_str_lit] / data[price_str_lit]
            print(f"Age {age}+: {ratio:.2f} pieces per dollar")
    
    print(f"\nBest value: Age {best_age}+ with {best_ratio:.2f} pieces per dollar")
    
    return age_groups
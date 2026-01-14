# src/motif_prediction_pipeline/regex_utils.py
import re
from itertools import product


def split_alternation_pattern(pattern):
    """
    Splits the top-level alternations in a regex pattern while respecting groups.
    """
    parts = []
    current_part = []
    depth = 0

    for char in pattern:
        if char == '(':
            depth += 1
            current_part.append(char)
        elif char == ')':
            depth -= 1
            current_part.append(char)
        elif char == '|' and depth == 0:
            parts.append(''.join(current_part))
            current_part = []
        else:
            current_part.append(char)

    if current_part:
        parts.append(''.join(current_part))

    return parts


def extract_components(option):
    """
    Extracts components from a regex pattern string.
    """
    components = []
    i = 0

    # Strip outer parentheses if it's a single group
    number_of_groups = 0
    for char in option:
        if char == '(': number_of_groups += 1

    if number_of_groups == 1:
        if option.startswith('(') and option.endswith(')'):
            option = option[1:-1]
        elif "|" not in option:
            option = option.replace("(", "").replace(")", "")

    while i < len(option):
        char = option[i]
        if char == '[':
            end = option.find(']', i)
            if end == -1: raise ValueError("Unmatched '[' in pattern")
            components.append(option[i:end + 1])
            i = end + 1
        elif char == '(':
            group_depth = 1
            end = i + 1
            while end < len(option) and group_depth > 0:
                if option[end] == '(':
                    group_depth += 1
                elif option[end] == ')':
                    group_depth -= 1
                end += 1
            components.append(option[i:end])
            i = end
        elif char in '{*+?':
            if char == '{':
                end = option.find('}', i)
                if end == -1: raise ValueError("Unmatched '{' in pattern")
                components.append(option[i:end + 1])
                i = end + 1
            else:
                components.append(char)
                i += 1
        else:
            components.append(char)
            i += 1
    return components


def flat_the_lsts(lst_of_tuples):
    big_lst = []
    flanking_lst = []
    for lst in lst_of_tuples:
        lst_of_lsts_components = []
        lst_of_flanking_status = []
        for l in lst:
            if isinstance(l, list):
                lst_of_lsts_components.extend(l)
                lst_of_flanking_status.extend([True] * len(l))
            else:
                lst_of_lsts_components.append(l)
                lst_of_flanking_status.append(False)
        big_lst.append(lst_of_lsts_components)
        flanking_lst.append(lst_of_flanking_status)
    return big_lst, flanking_lst


def handle_subgroups(component_lst):
    expanded_components = []
    for component in component_lst:
        if re.match(r'\{(\d+),(\d+)\}', component):
            min_count, max_count = map(int, re.findall(r'\d+', component))
            previous_component = ''.join(expanded_components.pop())
            expanded_variants = [[previous_component] * i for i in range(min_count, max_count + 1)]
            expanded_components.append(expanded_variants)
        elif re.match(r'\{(\d+)}', component):
            match = re.match(r'\{(\d+)\}', component)
            count = int(match.group(1))
            previous_component = ''.join(expanded_components.pop())
            expanded_variants = [[previous_component] * count]
            expanded_components.append(expanded_variants)
        else:
            expanded_components.append([component])
    possible_patterns_lst, flanking_lst = flat_the_lsts(list(product(*expanded_components)))
    return possible_patterns_lst, flanking_lst


def generate_regex_patterns(components):
    expanded_components = []
    for component in components:
        if component.startswith('(') and component.endswith(')'):
            splitted_component = split_alternation_pattern(component[1:-1])
            component_possibilities = []
            for sp in splitted_component:
                smaller_component = extract_components(sp)
                if len(smaller_component) == 1:
                    component_possibilities.extend(smaller_component)
                    continue
                generated_lsts, flanking_sub_lst = handle_subgroups(smaller_component)
                component_possibilities.extend(generated_lsts)
            expanded_components.append(component_possibilities)
        elif re.match(r'\{(\d+),(\d+)\}', component):
            min_count, max_count = map(int, re.findall(r'\d+', component))
            previous_component = ''.join(expanded_components.pop())
            expanded_variants = [[previous_component] * i for i in range(min_count, max_count + 1)]
            expanded_components.append(expanded_variants)
        elif re.match(r'\{(\d+)}', component):
            match = re.match(r'\{(\d+)\}', component)
            count = int(match.group(1))
            previous_component = ''.join(expanded_components.pop())
            expanded_variants = [[previous_component] * count]
            expanded_components.append(expanded_variants)
        else:
            expanded_components.append([component])

    possible_patterns_lst, flanking_lst = flat_the_lsts(list(product(*expanded_components)))
    possible_patterns = [''.join(pattern) for pattern in possible_patterns_lst]
    combined_patterns = list(zip(possible_patterns_lst, possible_patterns, flanking_lst))
    return combined_patterns


def analyze_detailed_sequence_with_complex_patterns(pattern, sequence):
    """
    Main function to analyze sequence against regex for key residues.
    """
    pattern = pattern.strip("$^")
    max_number_of_aa = 20
    threshold_aa = 5
    split_patterns = split_alternation_pattern(pattern)
    match_found = False
    found_regex_pattern = None
    details = []

    for option in split_patterns:
        option_details = []
        components = extract_components(option)
        number_of_flanking_region = 0
        for component in components:
            if '{' in component: number_of_flanking_region += 1

        if number_of_flanking_region > 0 or "|" in option or "(" in option:
            possible_regex_patterns = generate_regex_patterns(components)
        else:
            flanking_lst = [False] * len(components)
            possible_regex_patterns = [(components, pattern, flanking_lst)]

        filtered_patterns = []
        for pattern_lst, regex_str, flanking_lst in possible_regex_patterns:
            try:
                alt_pattern = re.compile(regex_str)
                alt_match = alt_pattern.match(sequence, 0)
                if alt_match:
                    if len(sequence) == len(pattern_lst):
                        filtered_patterns.append((pattern_lst, regex_str, flanking_lst))
                        found_regex_pattern = regex_str
                        break
                    elif "$" in pattern and len(sequence) + 1 == len(pattern_lst):
                        filtered_patterns.append((pattern_lst, regex_str, flanking_lst))
                        found_regex_pattern = regex_str
                        break
            except re.error:
                continue

        if not filtered_patterns:
            continue
        else:
            match_found = True

        final_component = filtered_patterns[0][0]
        final_cleared_component = [x for x in final_component if x != ""]

        for i, char in enumerate(sequence):
            if i >= len(final_cleared_component): break
            component = final_cleared_component[i]
            flanking = filtered_patterns[0][2][i]
            current_pos = i

            # Logic to identify Key Residue
            is_key_residue = False

            if re.match(r'\[[A-Z]+\]', component):
                aa_options = set(component[1:-1])
                if len(aa_options) < threshold_aa: is_key_residue = True
            elif re.match(r'\[\^[A-Z]+\]', component):
                # Negated class usually large -> not key?
                # Your logic: max_number_of_aa - len(excluded) < threshold
                excluded = set(component[2:-1])
                if (max_number_of_aa - len(excluded)) < threshold_aa: is_key_residue = True
            elif re.match(r'[A-Z]', component):
                # Literal
                is_key_residue = True
            elif component == '.':
                is_key_residue = False

            option_details.append({
                "group": component, "match": char, "position": current_pos,
                "is_flanking": flanking, "is_key_residue": is_key_residue
            })

        details.append(option_details)
        break  # Take first matching option

    return {
        "match_found": match_found,
        "found_regex_pattern": found_regex_pattern,
        "details": details
    }


def get_key_residue_indices(regex, matched_sequence, start_pos):
    """
    Wrapper to get absolute positions of key residues.
    """
    result = analyze_detailed_sequence_with_complex_patterns(regex, matched_sequence)
    key_indices = []

    if result['match_found'] and result['details']:
        for res in result['details'][0]:
            if res['is_key_residue']:
                # res['position'] is 0-based relative to match
                key_indices.append(start_pos + res['position'])

    return key_indices
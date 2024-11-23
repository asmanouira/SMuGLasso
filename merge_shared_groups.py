def merge_multiple_LD_groups(group_lists):
    """
    Merge group identifiers from multiple population lists into a unified list of shared group indices.

    Args:
        group_lists (list of lists): A list of group lists (e.g., [pop1, pop2, pop3, ...]).
    
    Returns:
        list: Merged list of shared group indices.
    """
    # Input validation
    if not group_lists or not all(len(lst) == len(group_lists[0]) for lst in group_lists):
        raise ValueError("All input group lists must be non-empty and of the same length.")
    
    num_lists = len(group_lists)
    num_elements = len(group_lists[0])
    
    # Initialize shared groups list
    shared_groups = [1]
    
    # Iterate through group identifiers
    for i in range(1, num_elements):
        # Gather group identifiers at the current index across all lists
        current_groups = [group_lists[k][i] for k in range(num_lists)]
        previous_groups = [group_lists[k][i-1] for k in range(num_lists)]
        
        # Check conditions
        if current_groups == previous_groups:
            # Groups are consistent across all lists, continue the same shared group
            shared_groups.append(shared_groups[-1])
        elif all(g == current_groups[0] for g in current_groups):
            # All lists agree on the current group identifier, start a new shared group if different from previous
            shared_groups.append(shared_groups[-1] if current_groups[0] == previous_groups[0] else shared_groups[-1] + 1)
        else:
            # Disagreement or transition; start a new shared group
            shared_groups.append(shared_groups[-1] + 1)
    
    return shared_groups

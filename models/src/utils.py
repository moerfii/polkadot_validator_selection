def set_era_range(test_era):
    test_era = int(test_era)
    range_required = range(test_era-10, test_era+1)
    list_of_eras = []
    list_of_eras.extend(range_required)
    return list_of_eras


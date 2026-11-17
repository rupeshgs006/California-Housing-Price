def analyze_correlation(data):
    correlation=data.corr(numeric_only=True)
    target_correlation=correlation["MedHouseVal"]
    target_correlation=target_correlation.sort_values(ascending=False)
    return target_correlation

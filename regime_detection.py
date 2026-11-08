def regime(output):
    return {key: output["statistics"][key] for key in ("trend", "volatility", "adx", "atrPercentile", "efficiency", "emaSlope", "bandWidth")}
 #=============================================================================
# Metrics tracking for thesis comparison
# =============================================================================

def compute_metrics(archive, test_data, train_data, elapsed_time):
    """Compute all relevant QD and portfolio metrics."""
    
    # === QD Metrics ===
    coverage = len(archive) / cfg['n_niches']  # % niches filled
    
    fitnesses = [v.fitness for v in archive.values()]
    qd_score = sum(fitnesses)  # sum of all fitnesses
    best_fitness = max(fitnesses)
    mean_fitness = np.mean(fitnesses)
    
    # === Diversity in Archive ===
    descs = np.array([v.desc for v in archive.values()])
    vol_range = descs[:, 0].max() - descs[:, 0].min()
    div_range = descs[:, 1].max() - descs[:, 1].min()
    
    # === Best Policy Evaluation ===
    best_key = max(archive.keys(), key=lambda k: archive[k].fitness)
    best_actor = archive[best_key].x
    
    # Get weights
    input_state = train_data.iloc[-20:].values.flatten().astype(np.float32)
    with torch.no_grad():
        raw = best_actor(torch.FloatTensor(input_state).unsqueeze(0)).numpy().flatten()
    raw = raw - np.max(raw)
    weights = np.exp(raw) / np.sum(np.exp(raw))
    
    # Portfolio metrics
    herfindahl = np.sum(weights ** 2)  # concentration measure
    effective_n = 1 / herfindahl       # effective number of assets
    _, sharpe = calculate_test_performance(test_data, weights)
    
    return {
        # QD metrics
        'coverage': coverage,
        'qd_score': qd_score,
        'best_fitness': best_fitness,
        'mean_fitness': mean_fitness,
        'n_niches': len(archive),
        
        # Archive diversity
        'vol_range': vol_range,
        'div_range': div_range,
        
        # Portfolio quality
        'sharpe': sharpe,
        'herfindahl': herfindahl,
        'effective_n': effective_n,
        
        # Compute
        'time_seconds': elapsed_time,
    }
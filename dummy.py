P_pizza = 0.3232
P_entree = 0.7448
P_grill = 0.5310
P_corner = 0.4115

P_PEGC = P_pizza * P_entree * P_grill * P_corner
P_PEG = P_pizza * P_entree * P_grill
P_PEC = P_pizza * P_entree * P_corner
P_PE = P_pizza * P_entree

total_prob = P_PEGC + P_PEG + P_PEC + P_PE

P_PEGC_normalized = P_PEGC / total_prob
P_PEG_normalized = P_PEG / total_prob
P_PEC_normalized = P_PEC / total_prob
P_PE_normalized = P_PE / total_prob

print("Normalized probabilities:")
print("Path a (PEGC):", P_PEGC_normalized)
print("Path b (PEG):", P_PEG_normalized)
print("Path c (PEC):", P_PEC_normalized)
print("Path d (PE):", P_PE_normalized)

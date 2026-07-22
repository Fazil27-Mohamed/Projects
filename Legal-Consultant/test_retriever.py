from utils.retreiver import retrieve

query = "What are Fundamental Rights?"

results = retrieve(query)

print("\nTop Matching Chunks:\n")

for i, chunk in enumerate(results, 1):
    print(f"\n========== Result {i} ==========\n")
    print(chunk)
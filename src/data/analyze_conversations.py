import json
import statistics
from collections import Counter

FILE_PATH = "data/processed/amazon_conversations.jsonl"

turn_counts = []
customer_turns = 0
agent_turns = 0

conversation_lengths = Counter()

examples = []

with open(FILE_PATH, "r", encoding="utf-8") as f:

    for line in f:
        conversation = json.loads(line)

        messages = conversation["messages"]

        turn_count = len(messages)

        turn_counts.append(turn_count)
        conversation_lengths[turn_count] += 1

        for message in messages:

            if message["inbound"]:
                customer_turns += 1
            else:
                agent_turns += 1

        if len(examples) < 5:
            examples.append(conversation)


print("=" * 70)
print("AMAZONHELP CONVERSATION STATISTICS")
print("=" * 70)

print(f"Total conversations:       {len(turn_counts):,}")
print(f"Customer messages:         {customer_turns:,}")
print(f"Agent messages:            {agent_turns:,}")

print(f"\nAverage turns:             {statistics.mean(turn_counts):.2f}")
print(f"Median turns:              {statistics.median(turn_counts):.2f}")
print(f"Minimum turns:             {min(turn_counts)}")
print(f"Maximum turns:             {max(turn_counts)}")

print("\nConversation length distribution:")

for length in sorted(conversation_lengths)[:15]:
    count = conversation_lengths[length]
    print(f"{length:2} turns: {count:,}")


print("\n" + "=" * 70)
print("SAMPLE CONVERSATIONS")
print("=" * 70)

for conversation in examples:

    print("\n" + "-" * 70)
    print(f"Conversation ID: {conversation['conversation_id']}")

    for message in conversation["messages"]:

        role = (
            "CUSTOMER"
            if message["inbound"]
            else "AMAZONHELP"
        )

        print(f"\n{role}:")
        print(message["text"])
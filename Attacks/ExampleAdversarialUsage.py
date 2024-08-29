from GenerateAdversarial import generate_attack, list_of_attacks

attacks = list_of_attacks()
for attack in attacks:
    print(f"The attack {attack} is the essay:")
    essay = generate_attack(attack)
    print(essay)
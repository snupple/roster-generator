# roster.py

import random

class RosterGenerator:
    def __init__(self, names):
        self.names = names

    def generate_roster(self, count):
        return random.sample(self.names, count)

# Example usage
if __name__ == '__main__':
    generator = RosterGenerator(['Alice', 'Bob', 'Charlie', 'David'])
    print(generator.generate_roster(2))
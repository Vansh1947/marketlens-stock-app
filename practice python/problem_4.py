# Get input from the user. It's good practice to store it in a descriptive variable.
# Let's ask for fruit names separated by commas.
fruit_input_string = input("Enter fruit names, separated by commas (e.g., apple,banana,orange): ")

# The input is a single string. We can split it into a list of fruits.
# fruit_list = fruit_input_string.split(',')

# Now, let's use a for loop to print each fruit.
print("\nHere are the fruits you entered:")
for fruit in fruit_input_string.split(','):
    print(fruit.strip()) # .strip() removes any leading/trailing whitespace from each fruit name
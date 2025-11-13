# converter.py
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

if __name__ == "__main__":
    choice = input("Convert from (C)elsius or (F)ahrenheit? ").strip().lower()
    temp = float(input("Enter temperature: "))

    if choice == 'c':
        print(f"{temp}°C = {celsius_to_fahrenheit(temp):.2f}°F")
    elif choice == 'f':
        print(f"{temp}°F = {fahrenheit_to_celsius(temp):.2f}°C")
    else:
        print("Invalid choice.")


        

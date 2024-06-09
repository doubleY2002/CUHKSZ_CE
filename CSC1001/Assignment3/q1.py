class flower:
    def __init__(self,name="name",petals=0,price=0.0):
        self.name=name
        self.petals=petals
        self.price=price
    def print_flower(self):
        print("The name of the flower:",self.name)
        print("The number of petals:",self.petals)
        print("The price of the flower:",self.price)

def main():
    name=input("Input the name of the flower>")

    petal_num=input("Input the number of petals>")
    while True:
        if petal_num.isdigit(): break
        petal_num=input("Input the right number of petals>")

    price=input("Input its price>")
    while True:
        try:
            float(price)
            break
        except:
            price=input("Input the right price>")
    Myflower=flower(name,petal_num,price)
    Myflower.print_flower()

main()
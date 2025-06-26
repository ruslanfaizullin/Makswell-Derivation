def Makswell():
    while True:
        text = int(input("Choose one option please (write a number):\n"
                         "1: Conditions under which the Maxwell speed distribution derivation is valid\n"
                         "2: Maxwell's speeds distribution function for certain temperature and molar mass (1 mol)\n"
                         "3: Calculate the ratio of Maxwell's speeds distribution for certain range\n"
                         "4: Animation of the certain number of molecules with certain temperature and molar mass\n"))

        if text == 1:
            import Conditions

        elif text == 2:
            import Plot

        elif text == 3:
            import Probability

        elif text == 4:
            import Animation

Makswell()

import math
import sys

def desenhar_coracao():
    # Cores ANSI para o terminal
    VERMELHO = "\033[91m"
    NEGRITO = "\033[1m"
    RESET = "\033[0m"

    # Equação clássica do coração: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0
    # Ajustando proporções para a grelha de carateres do terminal
    linhas = []
    for y in range(15, -16, -1):
        linha = []
        for x in range(-30, 31):
            x_val = x * 0.05
            y_val = y * 0.1
            f = (x_val**2 + y_val**2 - 1)**3 - (x_val**2) * (y_val**3)
            if f <= 0.0:
                linha.append(f"{VERMELHO}{NEGRITO}*{RESET}")
            else:
                linha.append(" ")
        linhas.append("".join(linha))
    
    print("\n".join(linhas))

if __name__ == "__main__":
    desenhar_coracao()

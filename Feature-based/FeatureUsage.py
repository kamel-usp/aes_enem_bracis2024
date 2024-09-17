from GeradorLinearRegressor import get_linear_regressor, get_dados_tratados
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error 

def calcular_div(notas1, notas2):
    div = 0
    for n1, n2 in zip(notas1,notas2):
        if abs(n1 - n2) > 80:
            div += 1
    return div/len(notas1)

def printar_resultados(y, y_hat):
    print(f"ACC: {accuracy_score(y, y_hat)}")
    print(f"RMSE: {mean_squared_error(y, y_hat, squared=False)}")
    print(f"QWK: {cohen_kappa_score(y, y_hat, weights='quadratic')}")
    print(f"DIV: {calcular_div(y, y_hat)}")
    print(f"Latex: {100*accuracy_score(y, y_hat):.2f} & {mean_squared_error(y, y_hat, squared=False):.2f} & {cohen_kappa_score(y, y_hat, weights='quadratic'):.2f} & {100*calcular_div(y, y_hat):.2f}")

def arrumar_notas(notas):
    referencia = [0, 40, 80, 120, 160, 200]
    novas_notas = []
    for n in notas:
        dif = float('inf')
        nota_certa = -1
        for r in referencia:
            if abs(r-n) < dif:
                dif = abs(r-n)
                nota_certa = r
        novas_notas.append(nota_certa)
    return novas_notas

for i in range(1, 6):
    print(f"Competencia {i}") 
    rl = get_linear_regressor(i)
    teste = get_dados_tratados(i)['teste']
    y = teste['competencia']
    outro_novo = teste.drop('competencia', axis=1)
    x = outro_novo.values
    notas = rl.predict(x)
    y_hat = arrumar_notas(notas)
    printar_resultados(y, y_hat)
    print("---------------")

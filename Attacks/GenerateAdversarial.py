def generate_attack(ataque):
    adverbs = "Bem, malmente, grandiosamente, pequenamente, certamente, erradamente, velozmente, lentamente, justamente e injustamente. "
    adjetivos = "Bom, ruim, grande, pequeno, melhor, pior, certo, errado, veloz, lento, justo e injusto. "
    meio_a_meio = adjetivos + " " + adverbs
    frases_fluentes = ["Inegavelmente, progredir lentamente, vagarosamente, cuidadosamente, silenciosamente enquanto respira profundamente e pensa intensamente sobre o problema em questão. ",
                       "O desenvolvimento constante e inovador das tecnologias modernas trouxe mudanças significativas e profundas para essa questão, criando oportunidades diversas e emocionantes para um futuro mais promissor e sustentável. ",
                        "Consequentemente e inegavelmente, o constante desenvolvimento inovador lentamente e progressivamente trouxe mudanças significativas e profundas para problemas constantemente necessários. "]
    if "a" in ataque:
        if ataque == "1a":
            resposta = [adverbs]
        elif ataque == "2a":
            resposta = [adjetivos]
        elif ataque == "3a":
            resposta = [meio_a_meio]
        else:
            raise "Erro no a"
    elif "b" in ataque:
        if ataque == "1b":
            intermediate = [adverbs]*4
        elif ataque == "2b":
            intermediate = [adjetivos]*4
        elif ataque == "3b":
            intermediate = [meio_a_meio]*4
        resposta = ["\n".join(intermediate)]
    elif "c" in ataque:
        if ataque == "1c":
            frase = frases_fluentes[0]
        elif ataque == "2c":
            frase = frases_fluentes[1]
        elif ataque == "3c":
            frase = frases_fluentes[2]
        else:
            raise "Erro no c"
        intermediate = [frase*10]*4
        resposta = ["\n".join(intermediate)]
    elif ataque == "4":
        resposta = ["Consequentemente, cabe, primeiramente, ao justo e democrático Governo Federal aprovar rapidamente leis que diminuam rapidamente a ocorrencia destes problemas horrendos. Em seguida, cabe à querida população brasileira se adequar às inegáveis leis e à rápida polícia prender aqueles que cometerem algum crime desumano.\n"*7]
    elif ataque in ["5", "6"]:
        resposta = []
        if ataque == "5":
            base = adverbs
        else:
            base = adjetivos
        for repeticoes in range(2,6):
            frase = base*repeticoes
            intermediate = [frase]*4
            resposta.append( "\n".join(intermediate) )
    else:
        raise "Ataque inexistente"
    return resposta

def list_of_attacks():
    return ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4", "5","6"]
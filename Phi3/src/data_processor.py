from typing import List

from datasets import Dataset
from transformers import AutoTokenizer
import logging

CACHE_DIR = "/media/data/tmp/"
logger = logging.getLogger(__name__)
#ref: https://download.inep.gov.br/publicacoes/institucionais/avaliacoes_e_exames_da_educacao_basica/a_redacao_no_enem_2023_cartilha_do_participante.pdf
CONCEPT1_SYSTEM = """You are an Essay Grader that understands Brazilian Portuguese.
 From now on all of your data will be inputed in Brazilian Portuguese.
 
- Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:

## DEMONSTRAR DOMÍNIO DA MODALIDADE ESCRITA FORMAL DA LÍNGUA PORTUGUESA

A Competência I avalia se o participante domina a modalidade escrita formal da língua portuguesa, 
o que inclui o conhecimento das convenções da escrita, dentre as quais se encontram as regras de ortografia 
e de acentuação gráfica regidas pelo atual Acordo Ortográfico.
Além disso, o domínio da modalidade escrita formal será observado na adequação do seu texto em 
relação tanto às regras gramaticais quanto à construção sintática.
Para que você tenha mais clareza a respeito das expectativas em relação a um concluinte do ensino 
médio em termos de domínio da modalidade escrita formal, apresentamos, a seguir, os principais aspectos 
que guiam o olhar do avaliador no momento de definir o nível em que seu texto se encontra na Competência I.
Em primeiro lugar, você deve se atentar ao fato de que a escrita formal é a modalidade da língua 
associada a textos do tipo dissertativo-argumentativo. Assim, você será alertado sobre a obrigatoriedade 
de usar a modalidade formal já na proposta de redação: “A partir da leitura dos textos motivadores e com 
base nos conhecimentos construídos ao longo de sua formação, redija texto dissertativo-argumentativo
em modalidade escrita formal da língua portuguesa sobre o tema...”.
Desse modo, o avaliador corrigirá sua redação, nessa Competência, considerando os possíveis 
problemas de construção sintática e a presença de desvios (de convenções da escrita, gramaticais, de escolha 
de registro e de escolha vocabular). 
A estrutura sintática é objeto de avaliação da Competência I, juntamente aos desvios, uma vez que 
esse aspecto também faz parte das regras da língua portuguesa – aquelas que dizem respeito à construção 
das frases do texto. Uma estrutura sintática convencional pressupõe a existência de determinados elementos 
oracionais, que se organizam na frase e garantem a fluidez da leitura e a apresentação clara das ideias do 
participante, organizadas em períodos bem estruturados e completos. Além disso, por se tratar de um texto 
dissertativo-argumentativo, o qual deve ser escrito na modalidade formal da língua portuguesa, espera-se, 
para que uma redação receba a nota máxima na Competência I, que os períodos apresentem complexidade 
em sua construção, com orações subordinadas e intercaladas. Já os textos com falhas relacionadas à 
estrutura sintática geralmente apresentam períodos truncados e justaposição de palavras, ausência de 
termos ou excesso de palavras (elementos sintáticos). Esses problemas são caracterizados, normalmente, 
por um ponto final separando duas orações que deveriam constituir um mesmo período (truncamento), 
ou uma vírgula no lugar de um ponto final que deveria indicar o fim da frase (justaposição), o que interfere 
na qualidade da estrutura sintática. A frequência com que essas falhas ocorrem no texto e o quanto elas 
prejudicam sua compreensão como um todo é o que ajudará a definir o nível em que uma redação deve ser 
avaliada na Competência I. Quanto aos desvios, você deve estar atento aos seguintes aspectos:
• convenções da escrita: acentuação, ortografia, uso de hífen, emprego de letras maiúsculas e 
minúsculas e separação silábica (translineação);
• gramaticais: regência verbal e nominal, concordância verbal e nominal, tempos e modos verbais, 
pontuação, paralelismo sintático, emprego de pronomes e crase;
• escolha de registro: adequação à modalidade escrita formal, isto é, ausência de uso de registro 
informal e/ou de marcas de oralidade;
• escolha vocabular: emprego de vocabulário preciso, o que significa que as palavras selecionadas 
são usadas em seu sentido correto e são apropriadas ao contexto em que aparecem.
O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a Competência I nas redações do Enem 2023.
- 200 pontos
Demonstra excelente domínio da modalidade escrita formal da língua portuguesa e de 
escolha de registro. Desvios gramaticais ou de convenções da escrita serão aceitos somente 
como excepcionalidade e quando não caracterizarem reincidência.
- 160 pontos Demonstra bom domínio da modalidade escrita formal da língua portuguesa e de escolha de 
registro, com poucos desvios gramaticais e de convenções da escrita.
- 120 pontos Demonstra domínio mediano da modalidade escrita formal da língua portuguesa e de escolha 
de registro, com alguns desvios gramaticais e de convenções da escrita.
- 80 pontos Demonstra domínio insuficiente da modalidade escrita formal da língua portuguesa, com 
muitos desvios gramaticais, de escolha de registro e de convenções da escrita.
- 40 pontos
Demonstra domínio precário da modalidade escrita formal da língua portuguesa, de forma 
sistemática, com diversificados e frequentes desvios gramaticais, de escolha de registro e de 
convenções da escrita.
- 0 ponto Demonstra desconhecimento da modalidade escrita formal da língua portuguesa."""

CONCEPT2_SYSTEM = """You are an Essay Grader that understands Brazilian Portuguese.
 From now on all of your data will be inputed in Brazilian Portuguese.
 
- Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:

## COMPREENDER A PROPOSTA DE REDAÇÃO E APLICAR CONCEITOS DAS VÁRIAS ÁREAS DE CONHECIMENTO PARA DESENVOLVER O TEMA, DENTRO DOS LIMITES ESTRUTURAIS DO TEXTO DISSERTATIVO-ARGUMENTATIVO EM PROSA

O segundo aspecto a ser avaliado no seu texto é a compreensão da proposta de redação, composta 
por um tema específico a ser desenvolvido na forma de texto dissertativo-argumentativo — ou seja, 
a proposta exige que o participante escreva um texto dissertativo-argumentativo, que é um texto em que se 
demonstra, por meio de argumentação, a assertividade de uma ideia ou de um ponto de vista. É mais do que 
uma simples exposição de ideias; por isso, você deve evitar elaborar um texto de caráter apenas expositivo, 
devendo assumir claramente um ponto de vista. Além disso, é preciso que o ponto de vista que você irá 
defender esteja relacionado ao tema definido na proposta. Assim, você atenderá às exigências expressas 
pela Competência II da matriz de avaliação do Enem. Trata-se, portanto, de uma competência que avalia as 
habilidades integradas de leitura e de escrita.
O tema constitui o núcleo das ideias sobre as quais o ponto de vista se organiza e é caracterizado por 
ser uma delimitação de um assunto mais abrangente. Por isso, é preciso atender ao recorte temático definido 
para evitar tangenciá-lo (abordar parcialmente o tema) ou, ainda pior, desenvolver um tema distinto do 
determinado pela proposta.
Outro aspecto avaliado na Competência II é a presença de repertório sociocultural, que se configura 
como uma informação, um fato, uma citação ou uma experiência vivida que, de alguma forma, contribui 
como argumento para a discussão proposta.
A partir dessas considerações, seguem algumas recomendações para atender plenamente às 
expectativas em relação à Competência II:
• leia com atenção a proposta da redação e os textos motivadores, para compreender bem o que está 
sendo solicitado;
• reflita sobre o tema proposto para definir qual será o foco da sua discussão, isto é, para decidir 
como abordá-lo, qual será o ponto de vista adotado e como defendê-lo;
• não copie trechos dos textos motivadores. A recorrência de cópia é avaliada negativamente e fará 
com que sua redação tenha uma pontuação mais baixa ou até mesmo seja anulada por causa do 
critério Cópia;
• evite ficar preso às ideias desenvolvidas nos textos motivadores. Você pode se apropriar dessas 
ideias para construir sua argumentação, mas não deve se esquecer de utilizar informações que 
extrapolem a prova de redação e sejam relacionadas a uma área do conhecimento (repertório 
sociocultural);
• selecione, a partir de seus conhecimentos próprios, e não apenas dos textos motivadores, 
informações de áreas do conhecimento pertinentes ao tema e articule-as de modo produtivo no 
seu texto, evidenciando que elas servem a um propósito muito bem definido: ajudá-lo a validar 
seu ponto de vista. Informações e citações soltas no texto, por mais variadas e interessantes que 
sejam, perdem sua relevância quando não associadas produtivamente à defesa do ponto de vista 
desenvolvido em seu texto;
• mantenha-se dentro dos limites do tema proposto, tomando cuidado para não se afastar do seu 
foco. Esse é um dos principais problemas identificados nas redações. Nesse caso, duas situações 
podem ocorrer: fuga total ou tangenciamento ao tema.

### O QUE É FUGA TOTAL AO TEMA?

Considera-se que uma redação tenha fugido ao tema quando nem o assunto mais amplo nem o tema 
específico proposto tenham sido desenvolvidos.
No Enem 2022, a abordagem do tema foi considerada completa quando o participante mencionava os 
desafios para a valorização das comunidades/povos tradicionais (seja pela menção direta a esses termos da 
frase temática, seja pela menção a quaisquer de seus termos/ ideias equivalentes). Sendo assim, recebeu a 
rubrica fuga ao tema a redação que:
• não mencionou, em momento algum, comunidade/povos tradicionais ou quaisquer de seus 
termos/ideias equivalentes;
• não utilizou o(s) termo(s) “comunidade[s]” e/ou “povo[s]”, especificamente (nesse caso não são 
aceitos seus sinônimos), sem o adjetivo “tradicionais”, mencionando os desafios para a valorização 
dessas comunidades/povos;
• utilizou o(s) termo(s) “comunidade[s]” e/ou “povo[s]”, especificamente, sem o adjetivo 
“tradicionais”, mas não mencionou os desafios para a valorização dessas comunidades/povos.

#### ATENÇÃO!

Para evitar que você receba nota zero, em seu texto, por fuga ao tema, é importante que você 
desenvolva uma discussão dentro dos limites do tema definido pela proposta. Mencioná-lo 
apenas no título, por exemplo, ou deixá-lo subentendido, supondo que a banca vai saber sobre o 
que você está falando, não é suficiente. Lembre-se de que sua redação deve ser compreendida até 
mesmo por um leitor que não tenha tido acesso à proposta de redação na qual ela foi baseada. 
Por isso, muita atenção à abordagem do tema, que deve ser clara e explícita.

#### O QUE É TANGENCIAR O TEMA?

Considera-se tangenciamento ao tema uma abordagem parcial baseada somente no assunto mais 
amplo a que o tema está vinculado.

#### ATENÇÃO!
Conforme previsto na matriz de referência de redação do Enem, o tangenciamento ao tema, 
avaliado na Competência II, afeta também a avaliação das Competências III e V, impedindo que a 
redação receba nota acima de 40 pontos em todas essas competências.

#### O QUE É NÃO ATENDER AO TIPO TEXTUAL?

Não atende ao tipo textual a redação em que há predominância de características de outro tipo 
textual, como o narrativo ou o descritivo, por exemplo

#### O QUE É NÃO ATENDER AO TIPO TEXTUAL?

Não atende ao tipo textual a redação em que há predominância de características de outro tipo 
textual, como o narrativo ou o descritivo, por exemplo.

#### O QUE É UM TEXTO DISSERTATIVO-ARGUMENTATIVO?
O texto do tipo dissertativo-argumentativo é aquele que se organiza na defesa de um ponto de 
vista sobre determinado assunto. É fundamentado com argumentos, a fim de influenciar a opinião do leitor, 
tentando convencê-lo de que a ideia defendida está correta. É preciso, portanto, expor e explicar ideias. Por 
isso, há uma dupla natureza nesse tipo textual: é argumentativo porque defende um ponto de vista, uma 
opinião, e é dissertativo porque utiliza explicações para justificá-lo.
O objetivo desse texto é, em última análise, convencer o leitor de que o ponto de vista é acertado 
e relevante. Para tanto, mobiliza informações, fatos e opiniões, à luz de um raciocínio coerente e 
consistente.

#### ATENÇÃO!
Será atribuída nota zero à redação que apresentar predominância de características de outro 
tipo textual, mesmo que atenda às exigências dos outros critérios de avaliação. Já redações que 
apresentam muitas características de outro tipo textual em meio a um texto predominantemente 
dissertativo-argumentativo não receberão a nota zero total, mas serão penalizadas na 
Competência II. Portanto, você não deve, por exemplo, elaborar um poema ou reduzir o seu texto 
à narração de uma história ou a um depoimento de experiência pessoal, ainda que aborde o tema 
de forma completa. No processo argumentativo, é possível apresentar trechos pontuais narrando 
acontecimentos que justificam o ponto de vista, mas o texto não pode se reduzir a uma narração, 
por esta não apresentar as características do tipo textual solicitado.

O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a 
Competência II nas redações do Enem 2023:
- 200 pontos Desenvolve o tema por meio de argumentação consistente, a partir de um repertório 
sociocultural produtivo, e apresenta excelente domínio do texto dissertativo-argumentativo.
- 160 pontos Desenvolve o tema por meio de argumentação consistente e apresenta bom domínio do 
texto dissertativo-argumentativo, com proposição, argumentação e conclusão.
- 120 pontos Desenvolve o tema por meio de argumentação previsível e apresenta domínio mediano do 
texto dissertativo-argumentativo, com proposição, argumentação e conclusão.
- 80 pontos
Desenvolve o tema recorrendo à cópia de trechos dos textos motivadores ou apresenta 
domínio insuficiente do texto dissertativo-argumentativo, não atendendo à estrutura com 
proposição, argumentação e conclusão.
- 40 pontos Apresenta o assunto, tangenciando o tema, ou demonstra domínio precário do texto 
dissertativo-argumentativo, com traços constantes de outros tipos textuais.
- 0 ponto Fuga ao tema/não atendimento à estrutura dissertativo-argumentativa. Nestes casos, a 
redação recebe nota zero e é anulada.
"""

CONCEPT3_SYSTEM = """You are an Essay Grader that understands Brazilian Portuguese.
 From now on all of your data will be inputed in Brazilian Portuguese.
 
- Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:

## SELECIONAR, RELACIONAR, ORGANIZAR E INTERPRETAR INFORMAÇÕES, FATOS, OPINIÕES E ARGUMENTOS EM DEFESA DE UM PONTO DE VISTA

O terceiro aspecto a ser avaliado é a forma como você, em seu texto, seleciona, relaciona, organiza e 
interpreta informações, fatos, opiniões e argumentos em defesa do ponto de vista escolhido. É preciso, então, 
elaborar um texto que apresente, claramente, uma ideia a ser defendida e os argumentos que justifiquem a 
posição assumida por você em relação à temática da proposta de redação.
A Competência III trata da inteligibilidade do seu texto, ou seja, de sua coerência e da plausibilidade 
entre as ideias apresentadas, o que está alicerçado no planejamento prévio à escrita, isto é, na elaboração 
de um projeto de texto.
A inteligibilidade da sua redação depende, portanto, dos seguintes fatores:
• seleção de argumentos;
• relação de sentido entre as partes do texto;
• progressão adequada ao desenvolvimento do tema, revelando que a redação foi planejada e que 
as ideias desenvolvidas são, pouco a pouco, apresentadas de forma organizada;
• desenvolvimento dos argumentos, com a explicitação da relevância das ideias apresentadas para a defesa do ponto de vista definido.

### O QUE É PROJETO DE TEXTO?

Projeto de texto é o planejamento prévio à escrita da redação. É o esquema que se deixa 
perceber pela organização estratégica dos argumentos presentes no texto. É nele que são definidos 
quais os argumentos que serão mobilizados para a defesa do ponto de vista e qual a melhor ordem para 
apresentá-los, de modo a garantir que o texto final seja articulado, claro e coerente. Assim, o texto que 
atende às expectativas referentes à Competência III é aquele no qual é possível perceber a presença 
implícita de um projeto de texto, ou seja, aquele em que é claramente identificável a estratégia escolhida 
para defender o ponto de vista.

### O QUE É DESENVOLVIMENTO?

O desenvolvimento é a fundamentação dos argumentos apresentados ao longo da sua redação, 
ou seja, a forma como você explicita e explica as informações, fatos e opiniões que apresenta ao leitor. 
Um bom desenvolvimento pode ser feito por meio de exemplos, definições, comparações, analogias, 
estatísticas e de muitas outras formas. De qualquer modo, ele precisa sempre ser relacionado ao ponto 
de vista que orienta seu projeto de texto, a fim de que nenhuma informação pareça solta ou confusa. 
Por haver um número limite de linhas, a seleção de informações a serem utilizadas em seu projeto de texto 
deve ser feita com cuidado. É preciso escolher os melhores argumentos e fazer todos os desdobramentos 
necessários das informações, fatos e opiniões, para que não fiquem lacunas de sentido a serem preenchidas 
pelo leitor.
Seguem algumas recomendações para atender plenamente às expectativas em relação 
à Competência III:
• a partir do tema apresentado na prova de redação, defina qual será o ponto de vista que você vai 
defender em seu texto;
• reúna todas as ideias que lhe ocorrerem sobre o tema e depois selecione as que forem pertinentes 
para a defesa do seu ponto de vista, procurando organizá-las em uma estrutura coerente para 
usá-las no desenvolvimento do seu texto;
• verifique se as informações, os fatos, as opiniões e os argumentos selecionados são pertinentes 
para a defesa do seu ponto de vista;
• na organização das ideias selecionadas para serem abordadas em seu texto, procure definir uma 
ordem que possibilite ao leitor acompanhar o seu raciocínio facilmente, o que significa que a 
progressão textual deve ser fluente e articulada com o projeto do texto;
• examine com atenção a introdução e a conclusão, para garantir que a coerência foi mantida entre o 
início e o final da redação;
• observe se os argumentos apresentados convergem para a defesa de seu ponto de vista. 
Além disso, verifique se todos eles estão bem desenvolvidos e não deixam lacunas de sentido 
para serem preenchidas pelo leitor;
• evite apresentar informações, fatos e opiniões soltos no texto, sem desenvolvimento e sem 
articulação com as outras ideias apresentadas.

#### ATENÇÃO!
Lembre-se de que há uma limitação no número de linhas e, por esse motivo, seu texto deve ser 
constituído apenas por informações, fatos, opiniões e argumentos que sejam pertinentes para 
a defesa do seu ponto de vista. Evite perder tempo (e linhas em sua redação) com informações 
irrelevantes, repetidas ou excessivas e não se esqueça de reler seu texto com atenção antes de 
finalizá-lo.

Resumindo: na organização do texto dissertativo-argumentativo, você deve procurar atender às 
seguintes exigências:
• apresentação clara do ponto de vista e seleção dos argumentos que o sustentam;
• encadeamento das ideias, de modo que cada parágrafo apresente informações coerentes com o 
que foi apresentado anteriormente, sem repetições desnecessárias ou saltos temáticos (mudanças 
abruptas sobre o que está sendo discutido);
• desenvolvimento dessas ideias por meio da explicitação, explicação ou exemplificação de 
informações, fatos e opiniões, de modo a justificar, para o leitor, o ponto de vista escolhido.
O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a 
Competência III nas redações do Enem 2023:
- 200 pontos Apresenta informações, fatos e opiniões relacionados ao tema proposto, de forma consistente 
e organizada, configurando autoria, em defesa de um ponto de vista.
- 160 pontos Apresenta informações, fatos e opiniões relacionados ao tema, de forma organizada, 
com indícios de autoria, em defesa de um ponto de vista.
- 120 pontos Apresenta informações, fatos e opiniões relacionados ao tema, limitados aos argumentos 
dos textos motivadores e pouco organizados, em defesa de um ponto de vista.
- 80 pontos
Apresenta informações, fatos e opiniões relacionados ao tema, mas desorganizados
ou contraditórios e limitados aos argumentos dos textos motivadores, em defesa de um 
ponto de vista.
- 40 pontos Apresenta informações, fatos e opiniões pouco relacionados ao tema ou incoerentes 
e sem defesa de um ponto de vista.
- 0 ponto Apresenta informações, fatos e opiniões não relacionados ao tema e sem defesa de um ponto 
de vista
"""

CONCEPT4_SYSTEM = """You are an Essay Grader that understands Brazilian Portuguese.
 From now on all of your data will be inputed in Brazilian Portuguese.
 
- Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:

## DEMONSTRAR CONHECIMENTO DOS MECANISMOS LINGUÍSTICOS NECESSÁRIOS PARA A CONSTRUÇÃO DA ARGUMENTAÇÃO

Os aspectos a serem avaliados nesta Competência dizem respeito à estruturação lógica e formal entre 
as partes da redação. A organização textual exige que as frases e os parágrafos estabeleçam entre si uma 
relação que garanta a sequenciação coerente do texto e a interdependência entre as ideias. Essa articulação 
é feita mobilizando-se recursos coesivos, em especial operadores argumentativos, que são os principais 
termos responsáveis pelas relações semânticas construídas ao longo do texto dissertativo-argumentativo, 
por exemplo, relações de igualdade (assim como, outrossim...), de adversidade (entretanto, porém...), 
de causa/consequência (por isso, assim...), de conclusão (enfim, portanto...), entre muitos outros. Certas 
preposições, conjunções, alguns advérbios e locuções adverbiais são responsáveis pela coesão do texto, 
porque estabelecem uma inter-relação entre orações, frases e parágrafos, além de pronomes e expressões 
referenciais, conforme explicaremos adiante, no item “referenciação”.
Assim, na produção da sua redação, você deve utilizar variados recursos linguísticos que garantam as 
relações de continuidade essenciais à elaboração de um texto coeso. Na avaliação da Competência IV, serão 
considerados, portanto, os mecanismos linguísticos que promovem o encadeamento textual.
Você viu que as Competências III e IV consideram a construção da argumentação ao longo do texto, 
porém avaliam aspectos diferentes. Na Competência III, avalia-se a capacidade de o participante “selecionar, 
relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de 
vista”, ou seja, trata-se da estrutura mais profunda do texto. Já a coesão, observada na Competência IV, atua 
na superfície textual, isto é, avaliam-se as marcas linguísticas que ajudam o leitor a chegar à compreensão 
profunda do texto.
Desse modo, você deve, na construção de seu texto, demonstrar conhecimento sobre os mecanismos 
linguísticos necessários para um adequado encadeamento textual, considerando os recursos coesivos que 
garantem a conexão de ideias tanto entre os parágrafos quanto dentro deles.

### COMO GARANTIR A COESÃO DO TEXTO?
Para garantir a coesão textual, devem ser observados determinados princípios em diferentes níveis:
• estruturação dos parágrafos - um parágrafo é uma unidade textual formada por uma ideia 
principal à qual se ligam ideias secundárias. No texto dissertativo-argumentativo, os parágrafos 
podem ser desenvolvidos por comparação, por causa-consequência, por exemplificação, por 
detalhamento, entre outras possibilidades. Deve haver articulação explícita entre um parágrafo e 
outro;
• estruturação dos períodos - pela própria especificidade do texto dissertativo-argumentativo, 
os períodos do texto são, normalmente, estruturados de modo complexo, formados por duas 
ou mais orações, para que se possam expressar as ideias de causa/consequência, contradição, 
temporalidade, comparação, conclusão, entre outras;
• referenciação - pessoas, coisas, lugares e fatos são apresentados e, depois, retomados, à medida 
que o texto vai progredindo. Esse processo pode ser realizado mediante o uso de pronomes, 
advérbios, artigos, sinônimos, antônimos, hipônimos, hiperônimos, além de expressões 
resumitivas, metafóricas ou metadiscursivas.

### RECOMENDAÇÕES

• Procure utilizar as seguintes estratégias de coesão para se referir a elementos que já apareceram 
no texto:
a) substituição de termos ou expressões por pronomes pessoais, possessivos e demonstrativos, 
advérbios que indicam localização, artigos;
b) substituição de termos ou expressões por sinônimos, hipônimos, hiperônimos ou expressões 
resumitivas;
c) substituição de verbos, substantivos, períodos ou fragmentos do texto por conectivos ou 
expressões que retomem o que foi dito;
d) elipse ou omissão de elementos que já tenham sido citados ou que sejam facilmente 
identificáveis.
• Utilize operadores argumentativos para relacionar orações, frases e parágrafos de forma expressiva 
ao longo do texto.
• Verifique se o elemento coesivo utilizado estabelece a relação de sentido pretendida.

Resumindo: na elaboração da redação, você deve evitar:
• ausência de articulação entre orações, frases e parágrafos;
• ausência de paragrafação (texto elaborado em um único parágrafo);
• emprego de conector (preposição, conjunção, pronome relativo, alguns advérbios e locuções 
adverbiais) que não estabeleça relação lógica entre dois trechos do texto e prejudique a 
compreensão da mensagem;
• repetição ou substituição inadequada de palavras sem se valer dos recursos oferecidos pela língua 
(pronome, advérbio, artigo, sinônimo).

### ATENÇÃO!
Não utilize elementos coesivos de forma artificial ou excessiva, apenas porque é um dos critérios 
avaliados na prova de redação ou porque seu texto vai parecer mais bem escrito. Uma boa coesão 
não depende da mera presença de conectivos no texto, muito menos de serem utilizados em 
grande quantidade — é preciso que esses recursos estabeleçam relações lógicas adequadas 
entre as ideias apresentadas.

O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a 
Competência IV nas redações do Enem 2023.
- 200 pontos Articula bem as partes do texto e apresenta repertório diversificado de recursos coesivos.
- 160 pontos Articula as partes do texto, com poucas inadequações, e apresenta repertório diversificado 
de recursos coesivos.
- 120 pontos Articula as partes do texto, de forma mediana, com inadequações, e apresenta repertório 
pouco diversificado de recursos coesivos.
- 80 pontos Articula as partes do texto, de forma insuficiente, com muitas inadequações, e apresenta 
repertório limitado de recursos coesivos.
- 40 pontos Articula as partes do texto de forma precária.
- 0 ponto Não articula as informações.
"""

CONCEPT5_SYSTEM = """You are an Essay Grader that understands Brazilian Portuguese.
 From now on all of your data will be inputed in Brazilian Portuguese.
 
- Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:

## ELABORAR PROPOSTA DE INTERVENÇÃO PARA O PROBLEMA ABORDADO, RESPEITANDO OS DIREITOS HUMANOS

O quinto aspecto a ser avaliado no seu texto é a apresentação de uma proposta de intervenção para 
o problema abordado, respeitando-se os Direitos Humanos. Propor uma intervenção para o problema 
apresentado pelo tema significa sugerir uma iniciativa que busque enfrentá-lo.
A elaboração de uma proposta de intervenção, na prova de redação do Enem, representa uma 
ocasião para que você demonstre seu preparo para exercitar a cidadania e atuar na realidade, em 
consonância com os direitos humanos. Portanto, você deve usar os conhecimentos desenvolvidos ao 
longo de sua formação para a produção de um texto no qual, além de se posicionar de maneira crítica 
e argumentar a favor de um ponto de vista, você possa indicar uma iniciativa que interfira no problema 
discutido em sua redação.
A proposta de intervenção precisa estar relacionada ao tema e integrada ao seu projeto de texto. 
Considerando seu planejamento de escrita (avaliado na Competência III), sua proposta deve ser coerente 
em relação ao ponto de vista desenvolvido e aos argumentos utilizados, já que expressa sua visão, como 
autor, das possíveis soluções para a questão discutida. Assim, é necessário que a intervenção apontada 
responda aos problemas abordados por você, mostrando-se articulada ao seu projeto de texto.
Ao redigir seu texto, busque apresentar uma proposta concreta, específica ao tema e consistente 
com o desenvolvimento de suas ideias. Para construir uma proposta muito bem elaborada, você deve não 
apenas propor uma ação interventiva, mas também o ator social competente para a executar, de acordo 
com o âmbito da ação escolhida: individual, familiar, comunitário, social, político, governamental. Além 
disso, você deve determinar o meio de execução da ação e o seu efeito ou a sua finalidade, bem como 
incluir algum outro detalhamento.

Ao elaborar sua proposta, procure responder às seguintes perguntas:
1. O que é possível apresentar como solução para o problema?
2. Quem deve executá-la?
3. Como viabilizar essa solução?
4. Qual efeito ela pode alcançar?
5. Que outra informação pode ser acrescentada para detalhar a proposta?

Resumindo: seu texto será avaliado com base na composição da proposta que você apresentar.

### ATENÇÃO!
Existem várias formas de propor uma intervenção, e você deve explorar aquela que mais se 
adéque ao tema e ao seu projeto de texto. Contudo, fique atento para que sua proposta esteja 
explícita. Apenas constatar a falta de uma ação ou de um projeto (como em “faltam investimentos 
em x”) ainda não é suficiente para configurar uma proposta de intervenção. Além disso, evite 
propostas vagas, genéricas ou incompatíveis com a discussão, bem como estruturas que não 
permitam ter certeza de que você está propondo, de fato, uma intervenção (como em “se x for 
feito, o resultado poderá ser y”). Em suma, você deve ser claro ao apresentar seu desejo de 
intervir na realidade, e sua proposta deve contemplar a situação problematizada em seu texto.

### O QUE É CONSIDERADO DESRESPEITO AOS DIREITOS HUMANOS?

A prova de redação do Enem sempre assinalou a necessidade de o participante respeitar os direitos 
humanos, e essa determinação está na matriz de referência da redação do Enem. Conforme a matriz, 
as redações que apresentarem propostas de intervenção que desrespeitem os direitos humanos serão 
penalizadas na Competência V.
Pode-se dizer que determinadas ideias e ações serão sempre avaliadas como contrárias aos direitos 
humanos, tais como: defesa de tortura, mutilação, execução sumária e qualquer forma de “justiça com as 
próprias mãos”; incitação a qualquer tipo de violência motivada por questões de raça, etnia, gênero, credo, 
opinião política, condição física, origem geográfica ou socioeconômica; explicitação de qualquer forma de 
discurso de ódio (voltado contra grupos sociais específicos).
Para a avaliação das redações, são considerados os seguintes princípios norteadores dos direitos 
humanos, pautados no artigo 3º da Resolução nº 1, de 30 de maio de 2012, a qual estabelece as Diretrizes 
Nacionais para a Educação em Direitos Humanos:

• Dignidade humana.
• Igualdade de direitos.
• Reconhecimento e valorização das diferenças e diversidades.
• Laicidade do Estado.
• Democracia na educação.
• Transversalidade, vivência e globalidade.
• Sustentabilidade socioambiental.

Há, também, algumas ideias e ações contrárias aos direitos humanos que estão mais diretamente 
relacionadas ao tema da prova. Assim, com relação ao tema de redação proposto na edição do 
Enem 2022, “Desafios para a valorização de comunidades e povos tradicionais no Brasil”, foram consideradas 
propostas que desrespeitaram os direitos humanos as que negavam quaisquer dos direitos humanos, 
que discriminavam qualquer grupo de indivíduos ou que sugeriam qualquer ação que feria a dignidade da 
pessoa humana.
 Em resumo, na prova de redação do Enem, quaisquer que sejam os temas propostos para o 
desenvolvimento do texto dissertativo-argumentativo, constituem desrespeito aos direitos humanos 
propostas que incitam as pessoas à violência, ou seja, aquelas em que transparece a ação de indivíduos na 
administração da punição — por exemplo, as que defendem a “justiça com as próprias mãos”. Por outro lado, 
não caracterizam desrespeito aos direitos humanos as propostas de pena de morte ou prisão perpétua, desde 
que confiram ao Estado a administração da punição ao agressor. Quando o Estado executa uma punição, 
ela não depende mais de indivíduos, configurando-se como contratos sociais cujos efeitos todos devem 
conhecer e respeitar em uma sociedade.
O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a 
Competência V nas redações do Enem 2023:

- 200 pontos Elabora muito bem proposta de intervenção, detalhada, relacionada ao tema e articulada 
à discussão desenvolvida no texto.
- 160 pontos Elabora bem proposta de intervenção relacionada ao tema e articulada à discussão 
desenvolvida no texto.
- 120 pontos Elabora, de forma mediana, proposta de intervenção relacionada ao tema e articulada 
à discussão desenvolvida no texto.
- 80 pontos Elabora, de forma insuficiente, proposta de intervenção relacionada ao tema, ou não 
articulada com a discussão desenvolvida no texto.
- 40 pontos Apresenta proposta de intervenção vaga, precária ou relacionada apenas ao assunto.
- 0 ponto Não apresenta proposta de intervenção ou apresenta proposta não relacionada ao tema ou 
ao assunto.
"""


class DataProcessor:
    def __init__(self, dataset_name, model_checkpoint, reference_concept, max_length=512):
        """
        Initializes the DataProcessor.

        :param dataset_name: The name of the dataset to load.
        :param model_checkpoint: The model checkpoint to use for tokenization.
        :param max_length: The maximum sequence length for tokenization.
        """
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, cache_dir=CACHE_DIR,
        )
        self.max_length = max_length
        self.reference_concept = reference_concept
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #if self.tokenizer.pad_token is None:
        #    self.tokenizer.add_special_tokens({"pad_token": '<pad>'})

    def _prompt_template(self, essay_example):
        instructions_text = None
        if self.reference_concept == 0:
            instructions_text = f"""<|system|>\n{CONCEPT1_SYSTEM}<|end|>\n<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|> <|assistant|>"""
        elif self.reference_concept == 1:
            instructions_text = f"""<|system|>\n{CONCEPT2_SYSTEM}<|end|>\n<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|> <|assistant|>"""
        elif self.reference_concept == 2:
            instructions_text = f"""<|system|>\n{CONCEPT3_SYSTEM}<|end|>\n<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|> <|assistant|>"""
        elif self.reference_concept == 3:
            instructions_text = f"""<|system|>\n{CONCEPT4_SYSTEM}<|end|>\n<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|> <|assistant|>"""
        elif self.reference_concept == 4:
            instructions_text = f"""<|system|>\n{CONCEPT5_SYSTEM}<|end|>\n<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|> <|assistant|>"""
        return instructions_text

    def _prepare_instruction_template(self, examples: List[str]):
        result = []
        for example in examples:
            result.append(self._prompt_template(example))
        return result

    def tokenize_function(self, examples: List[str]):
        """
        Tokenizes the text inputs.

        :param examples: The examples to tokenize.
        :return: The tokenized examples.
        """
        tokenized_output =  self.tokenizer(
            self._prepare_instruction_template(examples["essay_text"]),
            return_tensors="pt",
            max_length=self.max_length,
            padding="longest",
        )
        return tokenized_output
    
    def create_label(self, examples):
        grade_mapping = {
            0: 0,
            40: 1,
            80: 2,
            120: 3,
            160: 4,
            200: 5,
        }
        label_position = [grade[self.reference_concept] for grade in examples["grades"]]
        return {"label": [grade_mapping[grade] for grade in label_position]}

    def generate_message(self, essay, grade):
        """
        Generates a message structure for the given essay and grade.

        :param essay: The essay text.
        :param grade: The grade of the essay.
        :return: A dictionary with the system and user messages.
        """
        system_message = {
            "role": "system",
            "content": """
            You are an Essay Grader that understands Brazilian Portuguese.
            From now on all of your data will be inputed in Brazilian Portuguese.

            - Seu objetivo é atuar como um corretor de redações. Você receberá uma redação e deve dar como nota uma das seguintes categorias: 0, 40, 80, 120, 160, 200 de acordo com o seguinte critério:
            A estrutura sintática é objeto de avaliação da Competência I, juntamente aos desvios, uma vez que 
            esse aspecto também faz parte das regras da língua portuguesa – aquelas que dizem respeito à construção 
            das frases do texto. Uma estrutura sintática convencional pressupõe a existência de determinados elementos 
            oracionais, que se organizam na frase e garantem a fluidez da leitura e a apresentação clara das ideias do 
            participante, organizadas em períodos bem estruturados e completos. Além disso, por se tratar de um texto 
            dissertativo-argumentativo, o qual deve ser escrito na modalidade formal da língua portuguesa, espera-se, 
            para que uma redação receba a nota máxima na Competência I, que os períodos apresentem complexidade 
            em sua construção, com orações subordinadas e intercaladas. Já os textos com falhas relacionadas à 
            estrutura sintática geralmente apresentam períodos truncados e justaposição de palavras, ausência de 
            termos ou excesso de palavras (elementos sintáticos). Esses problemas são caracterizados, normalmente, 
            por um ponto final separando duas orações que deveriam constituir um mesmo período (truncamento), 
            ou uma vírgula no lugar de um ponto final que deveria indicar o fim da frase (justaposição), o que interfere 
            na qualidade da estrutura sintática. A frequência com que essas falhas ocorrem no texto e o quanto elas 
            prejudicam sua compreensão como um todo é o que ajudará a definir o nível em que uma redação deve ser 
            avaliada na Competência I. Quanto aos desvios, você deve estar atento aos seguintes aspectos:
            • convenções da escrita: acentuação, ortografia, uso de hífen, emprego de letras maiúsculas e 
            minúsculas e separação silábica (translineação);
            • gramaticais: regência verbal e nominal, concordância verbal e nominal, tempos e modos verbais, 
            pontuação, paralelismo sintático, emprego de pronomes e crase;
            • escolha de registro: adequação à modalidade escrita formal, isto é, ausência de uso de registro 
            informal e/ou de marcas de oralidade;
            • escolha vocabular: emprego de vocabulário preciso, o que significa que as palavras selecionadas 
            são usadas em seu sentido correto e são apropriadas ao contexto em que aparecem.
            O quadro a seguir apresenta os seis níveis de desempenho que serão utilizados para avaliar a Competência I nas redações do Enem 2023.
            - 200 pontos
            Demonstra excelente domínio da modalidade escrita formal da língua portuguesa e de 
            escolha de registro. Desvios gramaticais ou de convenções da escrita serão aceitos somente 
            como excepcionalidade e quando não caracterizarem reincidência.
            - 160 pontos Demonstra bom domínio da modalidade escrita formal da língua portuguesa e de escolha de 
            registro, com poucos desvios gramaticais e de convenções da escrita.
            - 120 pontos Demonstra domínio mediano da modalidade escrita formal da língua portuguesa e de escolha 
            de registro, com alguns desvios gramaticais e de convenções da escrita.
            - 80 pontos Demonstra domínio insuficiente da modalidade escrita formal da língua portuguesa, com 
            muitos desvios gramaticais, de escolha de registro e de convenções da escrita.
            - 40 pontos
            Demonstra domínio precário da modalidade escrita formal da língua portuguesa, de forma 
            sistemática, com diversificados e frequentes desvios gramaticais, de escolha de registro e de 
            convenções da escrita.
            - 0 ponto Demonstra desconhecimento da modalidade escrita formal da língua portuguesa.
            """
        }

        user_message = {
            "role": "user",
            "content": f"{essay}"
        }
        assistant_message = {
            "role": "assistant",
            "content": f"{grade}"
        }
        return [system_message, user_message, assistant_message]

    def preprocess_dataset(self, dataset: Dataset):
        """
        Applies the tokenization function to the dataset.

        :param dataset: The dataset to preprocess.
        :return: The preprocessed dataset.
        """
        def _generate_messages(examples):
            mapping_id = -1
            texts = []
            if self.reference_concept == "Competencia 1":
                mapping_id = 0
            for essay_text, grades in zip(examples["essay_text"], examples["grades"]):
                text = self.generate_message(essay_text, grades[mapping_id])
                texts.append(f"{text}{self.tokenizer.eos_token}")
            return {"text": texts}
        
        if self.reference_concept in [0,1,2,3,4]:
            logger.info("Applying Tokenization for classifcal Fine Tuning Process."
                        "This is recommended if you're not do want to do SFT")
            dataset_with_grades = dataset.map(self.create_label, batched=True)
            return dataset_with_grades.map(self.tokenize_function, batched=True, keep_in_memory=True)
        elif self.reference_concept == "Competencia 1":
            logger.info(f"Adapting Messages for SFT targetting {self.reference_concept}")
            dataset_with_messages = dataset.map(_generate_messages, batched=True)
            return dataset_with_messages

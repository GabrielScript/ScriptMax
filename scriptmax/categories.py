"""As 4 categorias de áudio e as instruções específicas de cada uma."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    PERSONAL_GROWTH = "desenvolvimento-pessoal"
    MEDIA = "filmes-series"
    ACADEMIC = "academico"
    WORK = "trabalho"


@dataclass(frozen=True)
class CategoryProfile:
    label: str
    library_folder: str
    whisper_hint: str
    rules: str


ACADEMIC_MATH_RULES = """### Conteúdo de exatas detectado
- Converta para LaTeX toda fórmula ditada por extenso, matrizes (\\begin{bmatrix}), sistemas (\\begin{cases}), derivadas, integrais e limites.
- Para cada exemplo resolvido pelo professor, registre o enunciado e cada passo da resolução."""

ACADEMIC_THEORY_RULES = """### Conteúdo teórico detectado
- Não force fórmulas: relações simples ficam no texto corrido.
- Registre argumentos, autores, correntes de pensamento, fatos históricos/legais e os exemplos discutidos."""

PROFILES: dict[Category, CategoryProfile] = {
    Category.PERSONAL_GROWTH: CategoryProfile(
        label="Desenvolvimento pessoal / Psicologia",
        library_folder="Desenvolvimento pessoal e Psicologia",
        whisper_hint="Conteúdo sobre desenvolvimento pessoal e psicologia",
        rules="""## CATEGORIA: DESENVOLVIMENTO PESSOAL / PSICOLOGIA
Objetivo: transformar o conteúdo em um guia de autoconhecimento aplicável.
Estrutura (omita seções sem conteúdo no áudio):
# Ideia central — a tese principal em 2 a 4 frases.
# Conceitos e teorias — cada conceito com definição clara; cite autores, estudos e abordagens (TCC, psicanálise, estoicismo...) SOMENTE se mencionados.
# Como funciona — os mecanismos psicológicos/comportamentais explicados (causa -> efeito).
# Histórias e exemplos — casos e metáforas contados, resumidos com a lição de cada um.
# Ferramentas práticas — exercícios e técnicas em passo a passo numerado.
# Perguntas para reflexão — 3 a 6 perguntas derivadas do conteúdo para autoaplicação.
# Plano de ação — hábitos e ações concretas que o próprio conteúdo sugere.
Regras:
- Diferencie o que o autor apresenta como evidência científica do que é opinião ou experiência pessoal dele.
- Não acrescente diagnósticos nem conselhos clínicos que não estejam no áudio.""",
    ),
    Category.MEDIA: CategoryProfile(
        label="Filmes / Séries / Documentários",
        library_folder="Filmes, Séries e Documentários",
        whisper_hint="Áudio de filme, série ou documentário",
        rules="""## CATEGORIA: FILMES / SÉRIES / DOCUMENTÁRIOS
Objetivo: registro completo da obra para estudo e consulta.
Comece com a linha: > ⚠️ Este relatório contém spoilers.
Estrutura (omita seções sem conteúdo no áudio):
# Ficha — título, tipo (filme/série/episódio/documentário), personagens ou entrevistados citados.
# Sinopse — a história/o tema em um parágrafo.
# Desenvolvimento — os acontecimentos em ordem cronológica, por atos ou blocos.
# Personagens e motivações — quem são, o que querem, como mudam (ficção).
# Fatos, dados e argumentos — afirmações, números, fontes e contrapontos apresentados (documentário).
# Temas e simbolismos — temas centrais e como a obra os trabalha.
# Diálogos e falas marcantes — citações literais em blockquote.
# Lições e reflexões — o que a obra ensina ou provoca.
Regras:
- Separe o que a obra mostra/afirma da sua interpretação; marque interpretações como "Análise:".
- Não complete a trama com conhecimento externo: registre só o que está no áudio.""",
    ),
    Category.ACADEMIC: CategoryProfile(
        label="Acadêmico / Conhecimento",
        library_folder="Acadêmico e Conhecimento",
        whisper_hint="Aula sobre",
        rules="""## CATEGORIA: ACADÊMICO / CONHECIMENTO
Objetivo: uma apostila de estudo completa da aula.
Estrutura:
- Tópicos na ordem em que foram ensinados: definição -> propriedades/teoremas -> exemplos.
- Seção final "Pontos de atenção para prova": o que o professor enfatizou, repetiu ou disse que cai em prova.
- Seção final "Glossário": termos técnicos novos com definição de uma linha.""",
    ),
    Category.WORK: CategoryProfile(
        label="Trabalho",
        library_folder="Trabalho",
        whisper_hint="Reunião ou treinamento de trabalho sobre",
        rules="""## CATEGORIA: TRABALHO
Identifique o tipo e siga a estrutura correspondente (omita seções vazias).
Se for REUNIÃO:
# Resumo executivo — 3 a 5 linhas com o essencial.
# Contexto e pauta
# Decisões tomadas — cada decisão com a justificativa dada.
# Itens de ação — tabela | Tarefa | Responsável | Prazo |; use "não informado" quando o áudio não disser.
# Riscos, bloqueios e dependências
# Pendências e perguntas em aberto
Se for TREINAMENTO, PALESTRA OU INSTRUÇÃO:
# Objetivo do treinamento
# Procedimentos — passo a passo numerado, com ferramentas, sistemas e parâmetros citados.
# Regras, políticas e boas práticas
# Erros comuns e como evitá-los
# Checklist final
Regras:
- Registre números, datas, valores e nomes de sistemas exatamente como ditos.""",
    ),
}


def profile_for(category: Category) -> CategoryProfile:
    return PROFILES[category]

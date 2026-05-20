import pandas as pd
import os
from pathlib import Path

def criar_modelo_dimensional():
    print("Iniciando modelagem dimensional (Star Schema)...")
    
    raiz = Path(__file__).parent.parent
    caminho_gold_input = raiz / "data" / "gold" / "surf_gold_data.csv"
    output_dir = raiz / "data" / "gold" / "tabelas_dimensionais"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not caminho_gold_input.exists():
        print("Arquivo Gold base não encontrado.")
        return

    df = pd.read_csv(caminho_gold_input)
    df['data_hora'] = pd.to_datetime(df['data_hora'])

    # Dimensão praias
    print("Criando Dimensão Praias...")
    d_praias = df[['pico_nome', 'pico_orientacao', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    d_praias['id_praia'] = d_praias.index + 1
    d_praias.to_csv(output_dir / "dim_praias.csv", index=False)

    # Dimensão condição
    print("Criando Dimensão Condição...")
    # Agrupamos as descrições únicas para reduzir redundância na fato
    d_condicao = df[['predicao_modelo', 'indice_aproveitamento', 'surfista_nivel', 'mar_tipo']].drop_duplicates().reset_index(drop=True)
    d_condicao['id_condicao'] = d_condicao.index + 1
    d_condicao.to_csv(output_dir / "dim_condicao_ia.csv", index=False)

    # Dimensão calendário
    print("Criando Dimensão Calendário...")
    datas_unicas = pd.to_datetime(df['data_hora'].dt.date.unique())
    d_calendario = pd.DataFrame({'data': datas_unicas})
    d_calendario['id_data'] = d_calendario['data'].dt.strftime('%Y%m%d').astype(int)
    d_calendario['ano'] = d_calendario['data'].dt.year
    d_calendario['mes'] = d_calendario['data'].dt.month
    d_calendario['dia'] = d_calendario['data'].dt.day
    d_calendario['dia_semana_nome'] = d_calendario['data'].dt.day_name()
    d_calendario.to_csv(output_dir / "dim_calendario.csv", index=False)

    # Fato previsão de surf
    print("Criando Tabela Fato...")
    # Join para trazer apenas os IDs para a Fato
    f_previsao = df.merge(d_praias, on=['pico_nome', 'pico_orientacao', 'lat', 'lon']) \
                   .merge(d_condicao, on=['predicao_modelo', 'indice_aproveitamento'])
    
    f_previsao['id_data'] = f_previsao['data_hora'].dt.strftime('%Y%m%d').astype(int)
    
    # Selecionamos apenas as métricas e as FKs (Foreign Keys)
    colunas_fato = [
        'id_data', 'id_praia', 'id_condicao', 'hora',
        'tamanho_onda', 'periodo_onda', 'direcao_onda', 
        'velocidade_vento', 'direcao_vento', 'potencia_onda'
    ]
    
    f_previsao_final = f_previsao[colunas_fato]
    f_previsao_final.to_csv(output_dir / "fato_previsao_surf.csv", index=False)

    print(f"Modelo Dimensional exportado para: {output_dir}")

if __name__ == "__main__":
    criar_modelo_dimensional()
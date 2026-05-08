import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib 
import os
from pathlib import Path

def treinar_modelo():
    caminho_script = Path(__file__).parent 
    raiz_projeto = caminho_script.parent
    
    caminho_silver = raiz_projeto / "data" / "silver" / "surf_silver_data.csv"
    pasta_modelos = raiz_projeto / "modelos"
 
    # Carregamento dos dados (camada Silver)
    if not caminho_silver.exists():
        print(f"Erro: Arquivo Silver não encontrado em {caminho_silver}")
        return
        
    df = pd.read_csv(caminho_silver)

    # Criação de Encoders para variáveis categóricas
    le_pico = LabelEncoder()
    df['pico_id'] = le_pico.fit_transform(df['pico_nome'])

    le_vento = LabelEncoder()
    df['vento_id'] = le_vento.fit_transform(df['vento_tipo'])

    # Treino do modelo
    features = ['tamanho_onda', 'periodo_onda', 'direcao_onda', 'velocidade_vento', 
                'direcao_vento', 'pico_id', 'vento_id', 'potencia_onda', 'mes', 'hora']
    
    X = df[features]
    y = df['target_regra_manual']
    
    print("Treinando o modelo Random Forest...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    acuracia = modelo.score(X, y)
    print(f"Treino concluído! Acurácia do modelo: {acuracia:.2%}")

    # Salvando o modelo e os encoders
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(modelo, pasta_modelos / 'modelo_surf.pkl')
    joblib.dump(le_pico, pasta_modelos / 'encoder_pico.pkl')
    joblib.dump(le_vento, pasta_modelos / 'encoder_vento.pkl')

    print(f"Modelo e Encoders salvos com sucesso em: {pasta_modelos}")

if __name__ == "__main__":
    treinar_modelo()
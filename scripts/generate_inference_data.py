"""
Script para gerar dados de teste para pipeline de inferência
Simula novos dados de equipamentos para testar o sistema de inferência
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_equipment_data(n_samples=100, failure_rate=0.04):
    """
    Gera dados sintéticos de equipamentos no formato AI4I 2020
    """
    logger.info(f"Gerando {n_samples} amostras de dados sintéticos...")
    
    np.random.seed(42)  # Para reproduzibilidade
    
    # Parâmetros baseados no dataset AI4I 2020 Type L
    data = []
    
    for i in range(n_samples):
        # Condições normais de operação
        air_temp = np.random.normal(298.1, 2.0)  # ~25°C
        process_temp = air_temp + np.random.normal(10.0, 1.5)  # Sempre maior que ar
        rotational_speed = np.random.normal(1538, 179)  # RPM normal
        torque = np.random.normal(40, 9)  # Torque normal
        tool_wear = np.random.randint(0, 253)  # Desgaste da ferramenta
        
        # Simula diferentes cenários
        scenario = np.random.choice(['normal', 'stress', 'wear'], p=[0.7, 0.2, 0.1])
        
        if scenario == 'stress':
            # Condições de stress - maior probabilidade de falha
            process_temp += np.random.normal(5, 2)
            torque += np.random.normal(15, 5)
            rotational_speed -= np.random.normal(100, 30)
            
        elif scenario == 'wear':
            # Alto desgaste - maior probabilidade de falha
            tool_wear = np.random.randint(200, 253)
            torque += np.random.normal(10, 3)
        
        # Determina se há falha (baseado nas condições)
        failure_prob = failure_rate
        if scenario == 'stress':
            failure_prob *= 3
        elif scenario == 'wear':
            failure_prob *= 2
            
        has_failure = np.random.random() < failure_prob
        
        # Se há falha, ajusta parâmetros para ser mais realista
        if has_failure:
            if np.random.random() < 0.3:  # Heat Dissipation Failure
                process_temp += np.random.normal(8, 2)
            elif np.random.random() < 0.3:  # Power Failure
                torque *= np.random.uniform(1.2, 1.5)
                rotational_speed *= np.random.uniform(0.7, 0.9)
            elif np.random.random() < 0.3:  # Overstrain Failure
                torque *= np.random.uniform(1.3, 1.6)
            elif np.random.random() < 0.3:  # Tool Wear Failure
                tool_wear = max(tool_wear, 200)
            # Random failures ficam como estão
        
        data.append({
            'Air temperature [K]': max(293, air_temp),  # Mínimo 20°C
            'Process temperature [K]': max(air_temp + 1, process_temp),  # Sempre > ar
            'Rotational speed [rpm]': max(1000, rotational_speed),  # RPM mínimo
            'Torque [Nm]': max(10, torque),  # Torque mínimo
            'Tool wear [min]': int(max(0, tool_wear)),  # Não negativo
            'Machine failure': int(has_failure),
            'Equipment_ID': f"EQ_{i+1:04d}",
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Scenario': scenario
        })
    
    df = pd.DataFrame(data)
    
    # Estatísticas dos dados gerados
    logger.info(f"Dados sintéticos gerados:")
    logger.info(f"  - Total de amostras: {len(df)}")
    logger.info(f"  - Falhas: {df['Machine failure'].sum()} ({df['Machine failure'].mean():.2%})")
    logger.info(f"  - Cenários: {dict(df['Scenario'].value_counts())}")
    
    return df

def create_inference_batch(df, batch_size=50, include_target=False):
    """
    Cria um lote de dados para inferência (sem target)
    """
    logger.info(f"Criando lote de inferência com {batch_size} amostras...")
    
    # Seleciona amostras aleatórias
    batch_df = df.sample(n=min(batch_size, len(df)), random_state=42).copy()
    
    # Remove colunas que não são features de entrada
    columns_to_remove = ['Machine failure', 'Equipment_ID', 'Timestamp', 'Scenario']
    if not include_target:
        inference_df = batch_df.drop(columns=[col for col in columns_to_remove if col in batch_df.columns])
    else:
        inference_df = batch_df.drop(columns=[col for col in columns_to_remove[1:] if col in batch_df.columns])
    
    logger.info(f"Lote de inferência criado: {inference_df.shape}")
    logger.info(f"Colunas: {list(inference_df.columns)}")
    
    return inference_df, batch_df

def save_inference_data(df, filename="new_data.csv", include_metadata=False):
    """
    Salva dados de inferência no formato esperado pelo pipeline
    """
    # Cria diretório se não existir
    os.makedirs('/opt/airflow/data/inference', exist_ok=True)
    
    filepath = f'/opt/airflow/data/inference/{filename}'
    df.to_csv(filepath, index=False)
    
    logger.info(f"Dados de inferência salvos em: {filepath}")
    
    # Salva metadados se solicitado
    if include_metadata:
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'samples_count': len(df),
            'columns': list(df.columns),
            'file_size_kb': os.path.getsize(filepath) / 1024,
            'description': 'Dados sintéticos para teste de inferência em lote'
        }
        
        metadata_path = f'/opt/airflow/data/inference/{filename.replace(".csv", "_metadata.json")}'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadados salvos em: {metadata_path}")
    
    return filepath

def main():
    """
    Função principal para gerar dados de teste
    """
    logger.info("Gerando dados de teste para pipeline de inferência...")
    
    try:
        # Gera dados sintéticos
        synthetic_data = generate_synthetic_equipment_data(n_samples=200, failure_rate=0.04)
        
        # Cria lote para inferência (sem target)
        inference_batch, full_batch = create_inference_batch(synthetic_data, batch_size=50, include_target=False)
        
        # Salva dados para inferência
        save_inference_data(inference_batch, "new_data.csv", include_metadata=True)
        
        # Salva dados completos para validação posterior
        save_inference_data(full_batch, "validation_data.csv", include_metadata=True)
        
        logger.info("✅ Dados de teste gerados com sucesso!")
        logger.info("📁 Arquivos criados:")
        logger.info("   - /opt/airflow/data/inference/new_data.csv (para inferência)")
        logger.info("   - /opt/airflow/data/inference/validation_data.csv (para validação)")
        
    except Exception as e:
        logger.error(f"Erro ao gerar dados: {e}")
        raise

if __name__ == "__main__":
    main() 
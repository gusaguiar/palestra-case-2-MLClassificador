"""
Script para carregamento e processamento inicial dos dados AI4I 2020
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Classe para carregamento dos dados AI4I 2020 para manutenção preditiva"""
    
    def __init__(self, data_path: str = "/opt/airflow/data/raw"):
        self.data_path = Path(data_path)
        
    def load_ai4i_data(self) -> pd.DataFrame:
        """
        Carrega o dataset AI4I 2020 Type L
        
        Returns:
            DataFrame com os dados carregados
        """
        # Tenta carregar dados Type L primeiro
        type_l_path = self.data_path / "ai4i2020_type_l.csv"
        
        if type_l_path.exists():
            logger.info("Carregando dados AI4I 2020 Type L...")
            data = pd.read_csv(type_l_path)
        else:
            # Se não existir, filtra do dataset original
            original_path = self.data_path / "ai4i2020.csv"
            
            if not original_path.exists():
                raise FileNotFoundError(f"Dataset AI4I 2020 não encontrado em {original_path}")
            
            logger.info("Filtrando dados Type L do dataset original...")
            data_full = pd.read_csv(original_path)
            data = data_full[data_full['Type'] == 'L'].copy()
            
            # Salva os dados filtrados
            data.to_csv(type_l_path, index=False)
            logger.info(f"Dados Type L salvos em {type_l_path}")
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessamento básico dos dados
        
        Args:
            data: DataFrame com dados brutos
            
        Returns:
            DataFrame preprocessado
        """
        # Cria cópia para não modificar original
        processed_data = data.copy()
        
        # Remove colunas não necessárias se existirem
        cols_to_remove = ['UDI', 'Product ID', 'Type']
        for col in cols_to_remove:
            if col in processed_data.columns:
                processed_data = processed_data.drop(columns=[col])
        
        # Renomeia colunas para formato mais amigável
        column_mapping = {
            'Air temperature [K]': 'air_temperature',
            'Process temperature [K]': 'process_temperature', 
            'Rotational speed [rpm]': 'rotational_speed',
            'Torque [Nm]': 'torque',
            'Tool wear [min]': 'tool_wear',
            'Machine failure': 'target'
        }
        
        processed_data = processed_data.rename(columns=column_mapping)
        
        # Verifica se todas as colunas esperadas estão presentes
        expected_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 
                        'torque', 'tool_wear', 'target']
        
        missing_cols = set(expected_cols) - set(processed_data.columns)
        if missing_cols:
            logger.warning(f"Colunas faltando: {missing_cols}")
        
        return processed_data
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """
        Gera resumo dos dados carregados
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            Dicionário com estatísticas
        """
        summary = {
            'total_samples': len(data),
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'failure_rate': data['target'].mean() if 'target' in data.columns else None,
            'target_distribution': data['target'].value_counts().to_dict() if 'target' in data.columns else None
        }
        
        return summary
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str = "/opt/airflow/data/processed/processed_data.csv"):
        """
        Salva dados processados
        
        Args:
            data: DataFrame processado
            output_path: Caminho de saída
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        logger.info(f"Dados processados salvos em: {output_path}")


def main():
    """Função principal"""
    
    # Inicializa loader
    loader = DataLoader()
    
    try:
        # Carrega dados
        logger.info("Iniciando carregamento dos dados AI4I 2020...")
        data = loader.load_ai4i_data()
        
        logger.info(f"Dados carregados: {len(data)} amostras")
        
        # Preprocessa
        logger.info("Preprocessando dados...")
        processed_data = loader.preprocess_data(data)
        
        # Gera resumo
        summary = loader.get_data_summary(processed_data)
        
        logger.info("=== RESUMO DOS DADOS ===")
        logger.info(f"Total de amostras: {summary['total_samples']}")
        logger.info(f"Shape: {summary['shape']}")
        logger.info(f"Taxa de falhas: {summary['failure_rate']:.4f}")
        logger.info(f"Distribuição do target: {summary['target_distribution']}")
        
        # Salva dados processados
        loader.save_processed_data(processed_data)
        
        logger.info("Carregamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante carregamento: {e}")
        raise


if __name__ == "__main__":
    main() 
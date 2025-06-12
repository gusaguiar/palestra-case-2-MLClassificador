import mlflow
from mlflow.tracking import MlflowClient

# Configura MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

print("=== VERIFICANDO MODELOS NO MLFLOW ===")

try:
    # Lista modelos registrados
    registered_models = client.search_registered_models()
    print(f"Modelos registrados: {len(registered_models)}")
    
    for model in registered_models:
        print(f"\nModelo: {model.name}")
        
        # Lista versões do modelo
        versions = client.search_model_versions(f'name="{model.name}"')
        print(f"Versões: {len(versions)}")
        
        for version in versions:
            print(f"  v{version.version}:")
            print(f"    Source: {version.source}")
            print(f"    Run ID: {version.run_id}")
            print(f"    Status: {version.status}")
            
            # Tenta listar artefatos do run
            try:
                artifacts = client.list_artifacts(version.run_id)
                print(f"    Artefatos: {[a.path for a in artifacts]}")
            except Exception as e:
                print(f"    Erro ao listar artefatos: {e}")
            
            # Verifica aliases (MLflow 3.0)
            try:
                model_details = client.get_model_version(model.name, version.version)
                print(f"    Tags: {model_details.tags}")
            except Exception as e:
                print(f"    Erro ao obter detalhes: {e}")

except Exception as e:
    print(f"Erro geral: {e}") 
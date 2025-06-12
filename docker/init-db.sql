-- Criar usuário e database para MLOps
CREATE USER mlops WITH ENCRYPTED PASSWORD 'mlops123';
CREATE DATABASE mlops OWNER mlops;
GRANT ALL PRIVILEGES ON DATABASE mlops TO mlops;

-- Criar usuário e database para Airflow  
CREATE USER airflow WITH ENCRYPTED PASSWORD 'airflow123';
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Conceder permissões adicionais
ALTER USER mlops CREATEDB;
ALTER USER airflow CREATEDB; 
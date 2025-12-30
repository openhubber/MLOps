pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/openhubber/MLOps.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t fraud-detector .'
            }
        }
        stage('Push to Private Registry') {
            steps {
                sh '''
                    REG=http://host.docker.internal:6000
                    docker tag fraud-detector $REG/fraud-detector:latest
                    docker push $REG/fraud-detector:latest
                   '''
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'docker run -d -p 8000:8080 mydockerhub/fraud-detector:latest'
            }
        }
    }
}

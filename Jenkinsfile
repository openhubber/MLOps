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
        stage('Push to Docker Registry') {
            steps {
                withDockerRegistry([credentialsId: 'docker-hub-credentials', url: 'http://localhost:6000']) {
                    sh 'docker tag fraud-detector openhubber/fraud-detector:latest'
                    sh 'docker push openhubber/fraud-detector:latest'
                }
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'docker run -d -p 8000:8080 mydockerhub/fraud-detector:latest'
            }
        }
    }
}

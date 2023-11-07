package com.neuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

public class main {
    //esse algoritimo usa peso e altura combinados para adivinhar o genero de uma pessoa
    // presumido que quanto mais peso e altura mais indicios de que seja um homem.

    public static void main( String[] args ) {
        main main = new main();
        main.trainAndPredict();
    }

    //esse metodo é utilizado para chamar as funções, e passar os dados para elas, assim treinando os Neurons
    public void trainAndPredict() {
        List<List<Integer>> data = new ArrayList<List<Integer>>();
        data.add(Arrays.asList(115, 66));
        data.add(Arrays.asList(175, 78));
        data.add(Arrays.asList(205, 72));
        data.add(Arrays.asList(120, 67));
        List<Double> answers = Arrays.asList(1.0,0.0,0.0,1.0);

        Network network500 = new Network(500);
        network500.train(data, answers);

        Network network1000 = new Network(1000);
        network1000.train(data, answers);

        System.out.println("");
        System.out.println(String.format("  Homem, 167, 73: network500: %.10f | network1000: %.10f", network500.predict(167, 73), network1000.predict(167, 73)));
        System.out.println(String.format("Mulher, 105, 67: network500: %.10f | network1000: %.10f", network500.predict(105, 67), network1000.predict(105, 67)));
        System.out.println(String.format("Mulher, 120, 72: network500: %.10f | network1000: %.10f", network500.predict(120, 72), network1000.predict(120, 72)));
        System.out.println(String.format("  Homem, 143, 67: network500: %.10f | network1000: %.10f", network500.predict(143, 67), network1000.predict(120, 72)));
        System.out.println(String.format(" Homem', 130, 66: network500: %.10f | network1000: %.10f", network500.predict(130, 66), network1000.predict(130, 66)));
    }


    class Network {

        // epocas
        int epochs = 0; //1000;
        Double learnFactor = null;
        List<Neuron> neurons = Arrays.asList(
                new Neuron(), new Neuron(), new Neuron(),     //nos do entrada da rede neural
                new Neuron(), new Neuron(),                   //nos que representam as camadas ocultas da rede neural
                new Neuron());                                // nos de saida da rede neural

        public Network(int epochs){
            // epochs numero de vezes que o algoritimo vai rodar sobre o dataset de aprendizagem
            this.epochs = epochs;
        }
        public Network(int epochs, Double learnFactor) {
            this.epochs = epochs;
            this.learnFactor = learnFactor;
        }


        //O método predict é responsável por fazer previsões com base nos inputs input1 e input2
        public Double predict(Integer input1, Integer input2){
            return neurons.get(5).compute(
                    //O sexto neurônio get(5) calcula um valor com base nas saídas dos neurônios get(4) e get(3).
                    neurons.get(4).compute(
                            neurons.get(2).compute(input1, input2),
                            neurons.get(1).compute(input1, input2)
                    ),
                    neurons.get(3).compute(
                            neurons.get(1).compute(input1, input2),
                            neurons.get(0).compute(input1, input2)
                    )
            );
        }

        // esse é o metodo Train, utilizado para treinar os nossos Neurons
        public void train(List<List<Integer>> data, List<Double> answers){
            //Esta variável será usada para acompanhar o menor valor de perda (loss) obtido durante o treinamento.
            Double bestEpochLoss = null;
            // O loop executa o treinamento ao longo de um número especificado de épocas (epoch)
            for (int epoch = 0; epoch < epochs; epoch++){
                // O neurônio selecionado para a epoch atual é modificado aleatoriamente utilizando o método mutate.
                Neuron epochNeuron = neurons.get(epoch % 6);
                epochNeuron.mutate(this.learnFactor);
                // Lista que sera utilizada para as previsões
                List<Double> predictions = new ArrayList<Double>();
                //Para cada conjunto de dados na lista data,
                // o método predict é chamado com os valores do primeiro e segundo elementos do conjunto de dados
                for (int i = 0; i < data.size(); i++){
                    predictions.add(i, this.predict(data.get(i).get(0), data.get(i).get(1)));
                }
                //O método Util.meanSquareLoss é usado para calcular a perda entre as previsões (predictions) e as respostas reais (answers),
                // armazenando o resultado em thisEpochLoss.
                Double thisEpochLoss = Util.meanSquareLoss(answers, predictions);

                //A cada 10 épocas (epoch % 10 == 0),
                // o método imprime informações sobre o progresso do treinamento,
                // incluindo o número da época, a melhor perda até agora (bestEpochLoss)
                // e a perda atual (thisEpochLoss).
                if (epoch % 10 == 0) System.out.println(String.format("Epoch: %s | bestEpochLoss: %.15f | thisEpochLoss: %.15f", epoch, bestEpochLoss, thisEpochLoss));

                if (bestEpochLoss == null){
                    bestEpochLoss = thisEpochLoss;
                    epochNeuron.remember();
                } else {
                    if (thisEpochLoss < bestEpochLoss){
                        bestEpochLoss = thisEpochLoss;
                        epochNeuron.remember();
                    } else {
                        epochNeuron.forget();
                    }
                }
            }
        }
    }

    //classe Neuron, classe essa que tem os pesos e as tendencia.
    // representa um neuronio artificial
    class Neuron {
        Random random = new Random();

        //Representa o valor anterior do viés (bias) do neurônio.
        private Double oldBias = random.nextDouble(-1, 1), bias = random.nextDouble(-1, 1);
        //Representam os valores antigos e atuais do peso 1
        public Double oldWeight1 = random.nextDouble(-1, 1), weight1 = random.nextDouble(-1, 1);
        // Representam os valores antigos e atuais do peso 2 associado ao segundo input.
        private Double oldWeight2 = random.nextDouble(-1, 1), weight2 = random.nextDouble(-1, 1);

        public String toString(){
            return String.format("oldBias: %.15f | bias: %.15f | oldWeight1: %.15f | weight1: %.15f | oldWeight2: %.15f | weight2: %.15f", this.oldBias, this.bias, this.oldWeight1, this.weight1, this.oldWeight2, this.weight2);
        }


        //Este método é usado para modificar aleatoriamente os parâmetros do neurônio.
        // Ele aceita um argumento opcional learnFactor, que pode ser usado para ajustar a magnitude da mutação.

        public void mutate(Double learnFactor){
            //Gera um número aleatório entre 0 e 2 (inclusive) para escolher qual parâmetro será modificado (0 para bias, 1 para weight1, 2 para weight2).
            int propertyToChange = random.nextInt(0, 3);

            //gera um valor aleatório de mudança (changeFactor) que é somado ao parâmetro escolhido.
            Double changeFactor = (learnFactor == null) ? random.nextDouble(-1, 1) : (learnFactor * random.nextDouble(-1, 1));

            //se for 0 adicionar na bias se 1 no peso 1 se 2 no peso 2
            if (propertyToChange == 0){
                this.bias += changeFactor;
            } else if (propertyToChange == 1){
                this.weight1 += changeFactor;
            } else {
                this.weight2 += changeFactor;
            };
        }

        //Este método restaura os parâmetros do neurônio para seus valores antigos (armazenados nas variáveis oldBias, oldWeight1, e oldWeight2).
        public void forget(){
            bias = oldBias;
            weight1 = oldWeight1;
            weight2 = oldWeight2;
        }

        //Este método atualiza os valores antigos (oldBias, oldWeight1, e oldWeight2) para que reflitam os valores atuais
        public void remember(){
            oldBias = bias;
            oldWeight1 = weight1;
            oldWeight2 = weight2;
        }

        //Este método calcula a saída do neurônio com base em dois inputs (input1 e input2).
        public double compute(double input1, double input2){
            //Realiza um cálculo de pré-ativação, que é uma combinação linear dos inputs pelos pesos e somado ao viés: (weight1 * input1) + (weight2 * input2) + bias.
            double preActivation = (this.weight1 * input1) + (this.weight2 * input2) + this.bias;
            double output = Util.sigmoid(preActivation);
            return output;
        }
    }

    class Util {
        public static double sigmoid(double in){
            return 1 / (1 + Math.exp(-in));
        }
        public static double sigmoidDeriv(double in){
            double sigmoid = Util.sigmoid(in);
            return sigmoid * (1 - in);
        }

        public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers){
            double sumSquare = 0;
            for (int i = 0; i < correctAnswers.size(); i++){
                double error = correctAnswers.get(i) - predictedAnswers.get(i);
                sumSquare += (error * error);
            }
            return sumSquare / (correctAnswers.size());
        }
    }
}

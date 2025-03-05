import timeit

from classification_algorithms import k_neighbors, decision_tree, naive_bayes_bernoulli, naive_bayes_gaussian, \
    nb_compare, bagging, random_forest, ada_boost, gradient_boost
from clear_dataset import clear_data as step_1
from feature_extration import feature_extraction_3_variant_of_clear_datasets as step_3, knn_vizualizer, \
    bag_of_words_vizualizer
from unused_data_delete import unused_data_delete as step_2

if __name__ == '__main__':
    start_time = timeit.default_timer()
    ## Наименование файла с очищенными данными
    fileName = "train_v1.csv"

    print("Первичная очистка текстовых данных начата")
    step_1()
    print("Очистка завершена\n")

    print("Удаление неиспользуемых текстов из датасета начато")
    step_2()
    print("Удаление завершено\n")

    print("Визуализация текстовых данных")
    bag_of_words_vizualizer()
    step_3()
    knn_vizualizer(fileName)
    print("Конец визуализации\n")

    print("Классификация методом k-ближайших соседей")
    k_neighbors(fileName)
    print("Классификация завершена\n")

    print("Классификация методом Деревьев решений")
    decision_tree(fileName)
    print("Классификация завершена\n")

    print("Классификация методом наивного Байеса")
    naive_bayes_bernoulli(fileName)
    naive_bayes_gaussian(fileName)
    nb_compare(fileName)
    print("Классификация завершена\n")

    print("Классификация с использованием бутстрапирования")
    bagging(fileName)
    print("Классификация завершена\n")

    print("Классификация методом случайного леса")
    random_forest(fileName)
    print("Классификация завершена\n")

    print("Классификация с использованием адаптивного бустинга")
    ada_boost(fileName)
    print("Классификация завершена\n")

    print("Классификация с использованием градиентного бустинга")
    gradient_boost(fileName)
    print("Классификация завершена")

    end_time = timeit.default_timer()
    print("Время выполнения:", end_time)

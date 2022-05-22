import timeit

from clear_dataset import clear_data as step_1
from classification_algorithms import k_neighbors, decision_tree, naive_bayes_bernoulli, naive_bayes_gaussian, \
    nb_compare, bagging, random_forest, ada_boost, gradient_boost
from feature_extration import bag_of_words_vizualizer as step_3, knn_vizualizer
from unused_data_delete import unused_data_delete as step_2

if __name__ == '__main__':
    start_time = timeit.default_timer()
    print("Первичная очистка текстовых данных начата")
    step_1()
    print("Очистка завершена\n")

    print("Удаление неиспользуемых текстов из датасета начата")
    step_2()
    print("Удаление завершено\n")

    print("Визуализация текстовых данных")
    step_3()
    knn_vizualizer()
    print("Конец визуализации\n")

    print("Классификация методом k-ближайших соседей")
    k_neighbors()
    print("Классификация завершена\n")

    print("Классификация методом Деревьев решений")
    decision_tree()
    print("Классификация завершена\n")

    print("Классификация методом наивного Байеса")
    naive_bayes_bernoulli()
    naive_bayes_gaussian()
    nb_compare()
    print("Классификация завершена\n")

    print("Классификация с использованием бутстрапирования")
    bagging()
    print("Классификация завершена\n")

    print("Классификация методом случайного леса")
    random_forest()
    print("Классификация завершена\n")

    print("Классификация с использованием адаптивного бустинга")
    ada_boost()
    print("Классификация завершена\n")

    print("Классификация с использованием градиентного бустинга")
    gradient_boost()
    print("Классификация завершена")

    print("Классификация с использованием стекинга")
    print("Классификация завершена\n")
    end_time = timeit.default_timer()
    print("Время выполнения:", end_time)

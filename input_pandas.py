
import pandas


def citire_file():
    cf = pandas.read_csv("septembrie2020_augmentat.csv")
    return cf

print(citire_file().head(10))

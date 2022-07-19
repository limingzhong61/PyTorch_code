import os
import pandas as pd


def csv2excel(csv_dir_path, out_excel_path):
    """
        将csv_dir_path下的csv文件名简单的存储在一个excel的一个表格的一列里
    :param csv_dir_path: csv目录路径
    :param out_excel_path: excel文件路径
    :return: void
    """
    csv_file_name_list = []
    excel_writer = pd.ExcelWriter(out_excel_path)
    for file_name in os.listdir(csv_dir_path):
        if file_name.endswith(".csv"):
            # csv_file_name_list.append(file_name.replace(".csv", ""))
            csv_df = pd.read_csv(csv_dir_path + "/" + file_name)
            # print(csv_df.head())
            csv_df.to_excel(excel_writer, sheet_name=file_name)
    excel_writer.save()


if __name__ == '__main__':
    csv_dir_path = './Dataset and DataLoader_8/titanic'
    out_excel_path = "./out_excel.xlsx"
    csv2excel(csv_dir_path, out_excel_path)

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import sys
import cv2 as cv
import numpy as np
import time
import imutils
import easyocr
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf

text_train = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х',
              'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'ы', 'Ь', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'к', 'л',
              'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'э', 'ю', 'я']
char_n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
          57, 58, 59, 60]


def sort_contours(cnts):
    reverse = False
    i = 1
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    arr = [boundingBoxes[0][1] + boundingBoxes[0][3] / 2]
    for x, y, w, z in boundingBoxes:
        flag = 0
        for i in arr:
            if y < i < y + z:
                flag += 1
        if flag == 0:
            arr.append(y + z / 2)
    arr = sorted(arr)
    cnts = list(cnts)
    boundingBoxes = list(boundingBoxes)
    for i in arr:
        flag = 1
        while flag != 0:
            flag = 0
            for j in range(len(boundingBoxes) - 1):
                if (boundingBoxes[j][1] < i < boundingBoxes[j][1] + boundingBoxes[j][3] and boundingBoxes[j][0] >
                        boundingBoxes[j + 1][0] and boundingBoxes[j + 1][1] < i < boundingBoxes[j + 1][1] +
                        boundingBoxes[j + 1][3]):
                    cnts[j], cnts[j + 1] = cnts[j + 1], cnts[j]
                    flag = 1
                    boundingBoxes[j], boundingBoxes[j + 1] = boundingBoxes[j + 1], boundingBoxes[j]

    cnts = tuple(cnts)
    # Сортировка по строкам

    return cnts, boundingBoxes


def get_letters(img):
    # Получение контуров слов

    letters = []

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10)
    dilated = cv.dilate(binary, None, iterations=3)
    cnts = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts)[0]

    for c in cnts:
        if cv.contourArea(c) > 10:

            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = binary[y:y + h, x:x + w]
            if roi.shape != binary.shape:
                letters.append(roi)

    return letters


def convert_litters(img, size):
    # Создание массива отдельных букв
    letters = []
    ret, binary = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            (x, y, w, h) = cv.boundingRect(cont)
            x, y, w, h = x - 1, y - 1, w + 2, h + 2
            cv.rectangle(binary, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = binary[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.zeros(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:

                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
            try:
                letters.append((x, w, cv.resize(letter_square, (size, size), interpolation=cv.INTER_AREA)))
            except:
                pass
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters


def learning():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(30, 30, 1),
                                            activation='relu'))
    model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(char_n), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def learn(letters, model, letters_ord, letters_test, letters_ord_test):
    t_start = time.time()
    y_test = letters_ord_test
    x_test = letters_test
    y_train = letters_ord
    x_train = letters
    x_train = np.reshape(x_train, (x_train.shape[0], 30, 30, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 30, 30, 1))

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train_for_ex = tf.keras.utils.to_categorical(y_train, len(char_n))
    y_test_for_ex = tf.keras.utils.to_categorical(y_test, len(char_n))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,
                                                                   factor=0.5, min_lr=0.00001)
    model.fit(x_train, y_train_for_ex, validation_data=(x_test, y_test_for_ex), callbacks=[learning_rate_reduction],
              epochs=30)
    print("Training done, dT:", time.time() - t_start)
    return model


def create_array_for_train(letters_for_train, letters_for_train_char):
    for i in range(1, 500):
        name = f'Data\\Train\\{i}.png'
        print(name)
        letter = get_letters(name)
        for j in range(len(letter)):

            letter[j] = cv.resize(letter[j], [30, 30])

            if j < len(char_n):
                letters_for_train.append(letter[j])
                letters_for_train_char.append(char_n[j])
    return letters_for_train, letters_for_train_char


def create_array_for_test(letters_for_test, letters_for_test_char):
    for i in range(400, 478):
        name = f'Data\\Test\\{i}.png'
        print(name)
        letter = get_letters(name)
        for j in range(len(letter)):

            letter[j] = cv.resize(letter[j], [30, 30])
            if j < len(char_n):
                letters_for_test.append(letter[j])
                letters_for_test_char.append(char_n[j])
    return letters_for_test, letters_for_test_char


def img_to_string(letters, s, model):
    for i in letters:
        img = np.reshape(i[2], (1, 30, 30, 1))
        predict = model.predict([img])
        result = np.argmax(predict, axis=1)
        s += text_train[result[0]]

    return s


def learn_processing():
    letters_for_train = []
    letters_for_train_char = []
    letters_for_test = []
    letters_for_test_char = []

    letters_for_train, letters_for_train_char = create_array_for_train(letters_for_train, letters_for_train_char)
    letters_for_test, letters_for_test_char = create_array_for_test(letters_for_test, letters_for_test_char)

    letters_for_test = np.asarray(letters_for_test)
    letters_for_train_char = np.asarray(letters_for_train_char)
    letters_for_test_char = np.asarray(letters_for_test_char)
    letters_for_train = np.asarray(letters_for_train)

    model = learning()
    model = learn(letters_for_train, model, letters_for_train_char, letters_for_test, letters_for_test_char)
    model.save('newbigemnist_letters.h5')


def startConvertRu(img):
    # learn_processing()
    model = tf.keras.models.load_model('newemnist_letters.h5')
    size = 30
    let = get_letters(img)
    s = ""
    for i in let:
        k = convert_litters(i, size)
        s = img_to_string(k, s, model)
        s += ' '

    return s


def converten(imgname):
    s = ''
    red = easyocr.Reader(['en'])
    res = red.readtext(imgname)

    for i in res:
        s += i[1]
        s += " "
    return s


def convert_cv_qt(cv_img):
    rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

    height, width, channel = rgb_image.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(rgb_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qImg)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = None
        self.filetype = None
        self.img = None
        self.pixmap = None
        self.filename = "1.jpg"
        self.setObjectName("MainWinow")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: rgb(194, 194, 194);")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setStyleSheet("#centralwidget {background-image: url(1.jpg);}")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalWidget.setStyleSheet("#horizontalWidget {background-image: url(1.jpg);}")
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButton.setStyleSheet("color: rgb(255, 255, 255);\n"
                                      "background-color: rgb(255, 102, 0);\n "
                                      "border-radius: 8px;\n"
                                      "font: 12pt \"MS Shell Dlg 2\";")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton, 0, QtCore.Qt.AlignTop)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.comboBox = QtWidgets.QComboBox(self.horizontalWidget)
        self.comboBox.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "background-color: rgb(255, 102, 0);\n "
                                    "font: 10pt \"MS Shell Dlg 2\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox, 0, QtCore.Qt.AlignTop)
        self.verticalLayout.addWidget(self.horizontalWidget)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(600, 800))
        self.label.setStyleSheet("#label {background-image: url(1.jpg);}")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("1.jpg"))
        self.label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label.setObjectName("label")
        self.label.adjustSize()
        self.horizontalLayout_7.addWidget(self.label)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "background-color: rgb(255, 102, 0);\n "
                                        "border-radius: 8px;\n"
                                        "font: 12pt \"MS Shell Dlg 2\";")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_2")
        self.pushButton_3.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "background-color: rgb(255, 102, 0);\n "
                                        "border-radius: 8px;\n"
                                        "font: 12pt \"MS Shell Dlg 2\";")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_2")
        self.pushButton_4.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "background-color: rgb(255, 102, 0);\n "
                                        "border-radius: 8px;\n"
                                        "font: 12pt \"MS Shell Dlg 2\";")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.horizontalLayout_7.addLayout(self.verticalLayout_2)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setStyleSheet("font: 15pt;\n"
                                       "background-color: rgb(255, 166, 107);\n"
                                       "color: rgb(0, 0, 0);")
        self.horizontalLayout_7.addWidget(self.textBrowser)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.setCentralWidget(self.centralwidget)
        self.translating(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def translating(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Добавьте Файл"))
        self.comboBox.setItemText(0, _translate("MainWindow", "ru"))
        self.comboBox.setItemText(1, _translate("MainWindow", "en"))
        self.pushButton_2.setText(_translate("MainWindow", "Конвертировать"))
        self.pushButton_3.setText(_translate("MainWindow", "Очистить"))
        self.pushButton_4.setText(_translate("Main Window", "Создать txt"))
        self.pushButton.clicked.connect(self.pushButton_Clicked)
        self.pushButton_2.clicked.connect(self.pushButton_2_Clicked)
        self.pushButton_3.clicked.connect(self.pushButton_3_Cliked)
        self.pushButton_4.clicked.connect(self.pushButton_4_Cliked)

    def pushButton_4_Cliked(self):
        if not (self.textBrowser.toPlainText().isspace()) and self.textBrowser.toPlainText() != "":
            self.filename, self.filetype = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                                 "Создать",
                                                                                 ".",
                                                                                 "Text Files(*.txt);;Word Files(*.docx);;"
                                                                                 "All Files(*)")
            try:
                file = open(f"{self.filename}", "w")
                file.write(self.textBrowser.toPlainText())
                file.close()
            except:
                pass

    def pushButton_3_Cliked(self):
        self.textBrowser.setText("")

    def handAct(self, index):
        print(self.comboBox.itemText(index))
        return self.comboBox.itemText(index)

    def pushButton_Clicked(self):
        try:

            try:
                self.filename = QtWidgets.QFileDialog.getOpenFileName()[0]
            except:
                pass
            self.pixmap = QtGui.QPixmap(self.filename)
            img = cv.imread(self.filename)
            self.pixmap = convert_cv_qt(img)
            print(self.pixmap)
            if 600 / self.pixmap.width() < 800 / self.pixmap.height():
                self.label.setPixmap(self.pixmap.scaledToWidth(600))
            else:
                self.label.setPixmap(self.pixmap.scaledToHeight(800))
            self.label.adjustSize()
        except:
            msg = QMessageBox()

            msg.setIcon(QMessageBox.Warning)
            msg.setText("Ошибка открытия")
            msg.setInformativeText('Не получилось открыть картинку или '
                                   'произошла попытка открытия файла типа, который не поддерживается')
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def pushButton_2_Clicked(self):
        try:
            if self.filename != "1.jpg":
                if self.comboBox.currentText() == 'ru':
                    self.img = cv.imread(self.filename)
                    self.textBrowser.setText(startConvertRu(self.img))
                else:
                    self.img = cv.imread(self.filename)
                    self.textBrowser.setText(converten(self.filename))
        except:
            mes = QMessageBox()
            mes.setIcon(QMessageBox.Warning)
            mes.setText("Что то пошло не так")
            mes.setWindowTitle("Ой")
            mes.setStandardButtons(QMessageBox.Ok)
            mes.exec_()


def Application():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("App.jpg"))
    window = Window()

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    Application()

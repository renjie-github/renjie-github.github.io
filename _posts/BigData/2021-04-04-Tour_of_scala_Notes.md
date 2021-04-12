---
layout: post
title: "Tour of Scala"
subtitle: "学习笔记"
author: "Roger"
header-img: "img/BigData/Scala.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Blog
---

# Tour of Scala分章要点笔记
## 为什么学习Scala
For Spark！Spark是用Scala语言开发的。与Java相比，可以少写很多冗余代码。

## 01 Scala语言特点
- 函数式
- 静态类型
- 可扩展
- 可与Java互操作（运行于JRE）

## 02 Scala数据类型
Scala中，Any是任何类型的父类型，其定义了特定的通用方法，如：equals，hashCode，toString。Any有两个直接的子类：AnyVal（用于表示值类型，如Double，Float，Unit，Int，……）和AnyRef（用于表示引用类型，如List，Option，及自己定义的class，对应于Java中的java.lang.Object）。  
与大多语言类似，类型转换是单向的。  
Nothing是任何类型的子类。没有类型为Nothing的值。一个常见应用是：非终止信号，如抛出异常、程序退出或无限循环（即，它是不计算得到一个值的表达式，或不正常返回的方法的类型）。

## 03 Classes
Scala的成员类型默认是public，使用private这一访问修改器（access modifier）可以让对应成员对其它类隐藏。其成员变量设置（setter）通过函数的方法进行，如设置变量x的值使用def x_= ...，成员变量的获取（getter）也是通过函数的方法进行，如获取变量x的值用def x = _x。示例如下：  
```Scala
class Point {
  private var _x = 0
  private var _y = 0
  private val bound = 100

  // getter
  def x = _x
  // setter, 在getter中对应的identifier（此处为'x'）后添加_=，后面跟参数
  def x_= (newValue: Int): Unit = {
    if (newValue < bound) _x = newValue else printWarning
  }

  def y = _y
  def y_= (newValue: Int): Unit = {
    if (newValue < bound) _y = newValue else printWarning
  }

  private def printWarning = println("WARNING: Out of bounds")
}

val point1 = new Point
point1.x = 99
point1.y = 101 // prints the warning
``` 
Scala中没有val或var的参数都有private参数，对外部类不可见。

## 04 默认参数值
类似于Java中使用overloaded method的方法，可以给函数变量设置默认值来实现可选参数。如果caller忽略了一个变量，那么随后的每个变量都需要指明name。  
如果从Java中调用Scala的代码，那么所有参数将不再是可选的。

## 05 Tuple
Scala中的Tuple包含固定数量的元素，每个元素可以有自己的数据类型，类型可以自动推断。Scala使用一系列类来表示tuple：Tuple2, Tuple3, ..., Tuple22。后面数字代表元素个数。  
Tuple中每个独立的元素被命名为：_1, _2, ...所以可以使用t._1等形式来访问其元素。Tuple中的元素可以通过模式匹配的方式（Tuple或Case）分离出来，其中Case的优势是具有named element，可读性更强。如：  
```Scala
// example 1
val ingredient = ("Sugar", 25)
println(ingredient._1, ingredient._2)
val (name, quatity) = ingredient

// example 2
val planets = List(
  ("Mercury, 57.9), ("Venus", 108.2), ("Earth", 149.6), ("Mars", 227.9)
)
planets.foreach{
  case ("Earth", distance) =>
    println(s"Our planet is $distant million kilometers from the sun")
  case _ =>
}
```

## 06 Trait
Trait被用于在各类之间共享接口（interface）及字段（field）。类似于Java 8中的接口。Class和Object可以扩展trait，但由于trait无法被实例化，所以trait没有参数。Trait在泛型类型和抽象方法中很有用，用'trait'关键词定义：  
```Scala
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A // 抽象类型A
}
```  
Trait的使用：用'extends'关键词扩展一个trait，然后使用'override'关键词实现其抽象方法。如： 
```Scala
class IntIterator(to: Int) extends Iterator[Int] {
  private var current = 0
  override def hasNext: Boolean = current < to
  override def next(): Int = {
    if (hasNext) {
      val t = current
      current += 1
      t
    } else 0
  }
}

val iterator = new IntIterator(10)
iterator.next() // returns 0
iterator.next() // returns 1
```
## 07 使用Mixin来构建class
Mixin是用来组成class的trait。Class值可以有一个superclass，但却可以有多个mixin。Superclass和mixin分别使用关键词'extends'和'with'来使用。Mixin和superclass必须有相同的supertype。  
```Scala
abstract class AbsIterator {
  type T // 抽象类型T
  def hasNext: Boolean
  def next(): T
}

class StringIterator(s: String) extends AbsIterator {
  type T = Char
  private var i = 0
  def hasNext = i < s.length
  def next() = {
    val ch = s charAt i
    i += 1
    ch
  }
}

// 因为RichIterator是trait，所以不需要去实现AbsIterator的抽象成员
trait RichIterator extends AbsIterator {
  // foreach的输入为一个函数f，函数f输入数据类型为泛型T，输出类型为Unit
  def foreach(f: T => Unit): Unit = while (hasNext) f(next())
}

class RichStringIter extends StringIterator("Scala") with RichIterator
val richStringIter = new RichStringIter
richStringIter.foreach(println)
```
## 08 Higher-order Functions
更高阶函数以别的函数作为输入并返回一个函数，这是因为函数在Scala中是一类值（first-class values）。在纯粹的面向对象编程中，一个好的实践是避免暴露以函数作为参数的方法，否则可能泄露对象的内部状态。泄露内部状态可能打破目标本身的不变性从而与封装相违背。  
以函数作为参数的一个典型例子是map，在Scala的collections中均有实现：  
```Scala
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(x => x * 2)
// 上面可以简化为:
val newSalaries = salaries.map(_ * 2)
```
也可以传递方法作为变量到higher-order functions中，因为编译器会将方法强转为函数。使用higher-order functions的一个原因是为了减少冗余代码。  
在某些应用场景会想要产生一个函数，由一个方法返回一个函数的例子是：
```Scala
// 该函数的返回类型为一个函数'(String, String) => String'，即将两个string作为输入，返回一个string
def urlBuilder(ssl: Boolean, domainName: String): (String, String) => String = {
  val schema = if (ssl) "https://" else "http://"
  (endpoint: String, query: String) => s"$schema$domainName/$endpoint?$query"
}

val domainName = "www.example.com"
def getURL = urlBuilder(ssl=true, domainName)
val endpoint = "users"
val query = "id=1"
val url = getURL(endpoint, query) // "https://www.example.com/users?id=1": String
```
# 09 嵌套方法
在Scala中可以嵌套方法（Nested Methods），如下为一个求阶的例子：  
```Scala
def factorial(x: Int): Int = {
  def fact(x: Int, accumulator: Int): Int = {
    if (x <= 1) accumulator
    else fact(x - 1, x * accumulator)
  }
  fact(x, 1)
}

println("Factorial of 3: " + factorial(3)) // 6
```
# 10 多参数列表
多参数列表（Multiple Parameter Lists）的典型用例是foldLeft：
```Scala
trait Iterable[A] {
  ...
  // foldLeft应用一个2参数函数op到一个初始值z及该集合的所有元素上
  def foldLeft[B](z: B)(op: (B, A) => B): B
  ...
}

val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
val res = numbers.foldLeft(0)((m, n) => m + n)
```
多参数列表的建议用例有：
- Drive Type Inference
  在Scala中，类型推断一次处理一个参数列表。
  ```Scala
  // example 1:
  def foldLeft1[A, B](as: List[A], b0: B, op: (B, A) => B) = ???
  // 错误调用用法，因为Scala仍在推断类型A和B，所以无法推断函数_ + _的类型。通过将参数op移动到它自己的参数列表，A和B将在第一个参数列表中被推断。这些推断的类型将可被第二个参数列表获得并且_ + _将匹配推断到的类型：(Int, Int) => Int
  def notPossible = foldLeft1(numbers, 0, _ + _)
  // 需要用以下方法调用
  def firstWay = foldLeft1[Int, Int](numbers, 0, _ + _)
  def secondWay = foldLeft1(numbers, 0, (a: Int, b: Int) => a + b)

  // example 2，更简洁的定义方法
  def foldLeft2[A, B](as: List[A], b0: B)(op: (B, A) => B) = ???
  def possible = foldLeft2(numbers, 0)(_ + _)
  ```
- Implicit Parameters
  为了将特定参数指定为隐式参数（implicit），必须将这些参数放在它们自己的implicit参数列表中：
  ```Scala
  def execute(arg: Int)(implicit ec: scala.concurrent.ExecutionContext) = ???
  ```
- Partial Application
  当以更少数量的参数列表调用一个方法时，会产生一个以缺失参数列表作为其变量的函数。这一应用被称作partial application：
  ```Scala
  val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  // 传入一个元素为Int的空列表作为第一个参数，第二个参数op为空。_为匿名函数占位参数
  val numberFunc = numbers.foldLeft(List[Int]()) _

  // op为向前面元素（此处为列表）xs添加x^2。:+代表append
  val squares = numberFunc((xs, x) => xs :+ x*x)
  println(squares) // List(1, 4, 9, 16, 25, 36, 49, 64, 81, 100)

  // op为向前面元素（此处为列表）xs添加x^3。:+代表append
  val cubes = numberFunc((xs, x) => xs :+ x*x*x)
  println(cubes)  // List(1, 8, 27, 64, 125, 216, 343, 512, 729, 1000)
  ```

# 11 Case类及模式匹配
## Case类
Case类比较适合构建immutable data，常用于模式匹配。定义一个最小case类需要关键词case class及一个参数列表（可以为空）。因为case类默认有apply方法，该方法负责对象的构建，所以新建一个case类不需要'new'。当用参数创建一个case类时，参数类型是'public val'的。    
```Scala
case class Book(isbn: String)

val frankenstein = Book("978-123456")
```
- 引用 Case类实例的比较是通过结构而不是引用：  
  ```Scala
  case class Message(str1: String, str2: String)
  
  val message1 = Message("test string1", "test string2")
  val message2 = Message("test string1", "test string2")
  val messageAreTheSame = message1 == message2 // true
  ```
- 拷贝 使用copy方法可以创建一个实例的浅拷贝  
  ```Scala
  case class Message(sender: String, recipient: String, body: String)
  val message4 = Message("julien@bretagne.fr", "travis@washington.us", "Me zo o komz gant ma   amezeg")
  val message5 = message4.copy(sender = message4.recipient, recipient = "claire@bourgogne.fr")
  message5.sender  // travis@washington.us
  message5.recipient // claire@bourgogne.fr
  message5.body  // "Me zo o komz gant ma amezeg"
  ```  
## 模式匹配
模式匹配是Java中'switch'的加强版。一个match expression包含一个value，match关键字，以及至少一个case子句。Match表达式具有返回值。  
```Scala
abstract class Notification
case class Email(sender: String, title: String, body: String) extends Notification
case class SMS(caller: String, message: String) extends Notification

def showNotification(nofitication: Notification, importantPeopleInfo: Seq[String]): String = {
  nofitication match {
    // if作为模式守卫（pattern guard），用于获取更明确的匹配
    case Email(sender, title, _) if importantPeopleInfo.contains(sender) => 
      s"You got an email from VIP: $sender with title: $title!"
    case SMS(number, message) => 
      s"You got an SMS from $number! Message: $message"
    case _ => "You received a message from Unknown source." // 匹配其它类型
  }
}
```  
### 按照类型匹配
```Scala
abstract class Device
case class Phone(model: String) extends Device {
  def screenOff = "Turning screen off"
}
case class Computer(model: String) extends Device {
  def screenSaverOn = "Turning screen saver on..."
}

def goIdle(device: Device) = device match {
  case p: Phone => p.screenOff // 习惯上使用类型的第一个字母作为标识符
  case c: Computer => c.screenSaverOn
}
```
### 封装类（Sealed Class）
Trait和Class可以被标记为'sealed'，代表所有的子类型（subtype）都必须在同一个文件内声明，这确保了所有的子类都是已知的。  
```Scala
sealed abstract class Furniture
case class Couch() extends Furniture
case class Chair() extends Furniture

def findPlaceToSit(piece: Furniture): String = piece match {
  case a: Couch => "Lie on the couch"
  case b: Chair => "Sit on the chair"
}
```
Scala的模式匹配语句对于匹配通过case类表示的代数类型最有用。Scala还允许独立于case类定义模式：在extractor对象中使用unapply方法。  

# 12 单例对象
单例对象（Singleton object）是**只具有一个实例**的**类**。像lazy val一样，它只有被引用时才会创建（lazily）。单例对象使用关键字'object'定义。  
一个带有和它同名class的对象叫做伴生对象（companion object），对应的类叫做对象的伴生类（companion class）。伴生类/对象必须定义在同一个文件中。一个伴生类或对象可以获取其伴生对象或类的私有变量。对于不特定于伴生类实例的方法和值，使用伴生对象。如：
```Scala
import scala.math._

case class Circle(radius: Double) {
  import Circle._
  // 'area'成员对每个实例是特定的
  def area: Double = calculateArea(radius)
}

object Circle {
  // calculateArea对于每个实例都是可获得的
  private def calculateArea(radius: Double): Double = Pi * pow(radius, 2.0)
}

val circle1 = Circle(5.0)
println(circle1.area)
```
伴生对象也可以包含构造方法：  
```Scala
class Email(val username: String, val domainName: String)

object Email {
  def fromString(emailString: String): Option[Email] = {
    emailString.split("@") match {
      case Array(a, b) => Some(new Email(a, b))
      case _ => None
    }
  }
}

val scalaCenterEmail = Email.fromString("scala.center@epfl.ch")
scalaCenterEmail match {
  case Some(email) => println(
    s"""Registered an email
    |Username: ${email.username}
    |Domain name: ${email.domainName}
    """.stripMargin
  )
  case None => println("Error: could not parse email")
}
```
Java中的static成员在Scala中作为伴生对象中的普通成员被构建。当从Java中使用伴生对象时，成员将用'static'修饰符定义在伴生类中。这称为静态转发（static forwarding）。即使你自己没有定义伴生类，也会发生这种情况。

# 13 正则表达式
```Scala
import scala.util.matching.Regex

// 匹配一个模式
val numberPattern: Regex = "[0-9]".r

numberPattern.findFirstMatchIn("awesomepassword") match {
  case Some(_) => println("Password OK")
  case None => println("Password must contain a number")
}

// 匹配多个group
val keyValPattern: Regex = "([0-9a-zA-Z- ]+): ([0-9a-zA-Z-#()/. ]+)".r

val input: String =
  """background-color: #A03300;
    |background-image: url(img/header100.png);
    |background-position: top center;
    |background-repeat: repeat-x;
    |background-size: 2160px 108px;
    |margin: 0;
    |height: 108px;
    |width: 100%;""".stripMargin

for (patternMatch <- keyValPattern.findAllMatchIn(input))
  println(s"key: ${patternMatch.group(1)} value: ${patternMatch.group(2)}")
```
# 14 Extractor Object
Extractor Object是具有'unapply'方法的对象。'apply'方法类似于一个构造器，使用输入变量构造一个对象。而'unapply'方法接收一个对象并且尝试将其转化为变量，常用于模式匹配即partial functions。
```Scala
import scala.util.Random

object CustomerID {
  def apply(name: String) = s"$name--${Random.nextLong}"

  def unapply(customerID: String): Option[String] = {
    val stringArray: Array[String] = customerID.split("--")
    if (stringArray.tail.nonEmpty) Some(stringArray.head) else None
  }
}

// apply方法从一个name创建一个CustomerID字符串。该句等同于CustomerID.apply("Sukyoung")
val customer1ID = CustomerID("Sukyoung")  // Sukyoung--23098234908
customer1ID match {
  // unapply进行逆向操作，返回name。该句等同于CustomerID.unapply(customer1ID)
  case CustomerID(name) => println(name)  // prints Sukyoung
  case _ => println("Could not extract a CustomerID")
}
```
'unapply'方法的返回类型选择：
- 如果只用于测试，返回Boolean。如case even()
- 如果返回单个类型T的sub-value，返回Option[T]
- 如果想返回若干个sub-values：T1,...,Tn。将它们组合在一个optional tuple中：Option[(T1,...,Tn)]  

有时返回值的数量不是固定的，此时可以用'unapplySeq'方法来定义extractor，该extractor返回一个Option[Seq[T]]。这些模式的常见用例包括使用case List(x, y, z) => 来解构一个List，以及使用正则表达式Regex来分解一个字符串，如：case r(name, remainingFields @ _*) =>。

# 15 For表达式
```Scala
def foo(n: Int, v: Int) = 
  for (i <- 0 until n; j <- 0 until n if i + j == v)
    yield (i, j) // 如果只是需要执行"side-effects"，可以忽略yield

foo(10, 10) foreach {
  case (i, j) => 
    println(s"($i, $j)")
}
```

# 16 泛型类及其子类的Variance
泛型类（Generic Class）是将类型作为参数的类，对于收集类很有用。
```Scala
class Stack[A] {
  private var elements: List[A] = Nil // Nil在这里是一个空列表，不同于null
  // 重新将elements分配给一个新的，通过将x拼接到elements前面得到的list
  def push(x: A): Unit = elements = x :: elements 
  def peek: A = elements.head
  def pop(): A = {
    val currentTop = peek
    elements = elements.tail
    currentTop
  }
}

class Fruit
class Apple extends Fruit
class Banana extends Fruit

val stack = new Stack[Fruit]
val apple = new Apple
val banana = new Banana
stack.push(apple) // 可以存入子类
stack.push(banana)
```

# 17 Variance
Variance是复杂类型的子类相关性及它们的组成类型的子类关系。Scala支持泛型类类型参数的variance annotation。在类型系统中使用variance使得我们可以在复杂类型之间建立直观的联系，而缺乏方差会限制类抽象的重用。  
```Scala
class Foo[+A] // A covariant class（协变类）
class Bar[-A] // A contravariant class（逆变类）
class Baz[A] // An invariant class（不变类）
```
## Covariance
类型参数为T的泛型类可以通过注释+T成为协变类。Scala标准库中有一个不可变（immutable）的泛型类：**sealed abstract class List[+A]**，其中类型参数A是协变的。协变意味着对于B是A的子类型（subtype），那么**List[B]**就是**List[A]**的子类型。这允许我们使用泛型来建立非常有用和直观的子类型关系。  
```Scala
abstract class Animal {
  def name: String
}
case class Cat(name: String) extends Animal
case class Dog(name: String) extends Animal

def printAnimalNames(animals: List[Animal]): Unit = 
  animals.foreach{
    animal => println(animal.name)
  }

val cats: List[Cat] = List(Cat("Whiskers"), Cat("Tom"))
val dogs: List[Dog] = List(Dog("Fido"), Dog("Rex"))

// prints: Whiskers, Tom
printAnimalNames(cats)

// prints: Fido, Rex
printAnimalNames(dogs)
```
## Contravariance
类型参数为T的泛型类可以通过注释-T成为逆变类。这一方式在类和与它相似（但与Covariance相反）的类型参数之间创建了一种子类型关系：对于某个类**class Writer[-A]**，使A逆变意味着对于A是B的子类型，那么**Writer[B]**就是**Writer[A]**的子类型。下面的例子中：Cat是Animal的子类型，那么对于实现了逆变类方法的Printer，Printer[Animal]是Printer[Cat]的子类型（子类Cat的Printer类应该知道，对应Animal类的Printer类的方法。反之则不然）。  
```Scala
abstract class Printer[-A] {
  def print(value: A): Unit
}

class AnimalPrinter extends Printer[Animal] {
  def print(animal: Animal): Unit = 
    println("The animal's name is: " + animal.name)
}
class CatPrinter extends Printer[Cat] {
  def print(cat: Cat): Unit = 
    println("The cat's name is: " + cat.name)
}
```
上例中，如果**Printer[Cat]**知道如何打印**Cat**，**Printer[Animal]**知道如何打印**Animal**，那么**Printer[Animal]**知道如何打印**Cat**也是合理的。而反过来则不成立，因为**Printer[Cat]**不知道如何打印**Animal**。因此，我们应该可以用**Printer[Animal]**来代替**Printer[Cat]**，通过使**Printer[A]**逆变可以做到这一点。
```Scala
def printMyCat(printer: Printer[Cat], cat: Cat): Unit = 
  printer.print(cat)

val catPrinter: Printer[Cat] = new CatPrinter
val animalPrinter: Printer[Animal] = new AnimalPrinter

printMyCat(catPrinter, Cat("Boots")) // The cat's name is: Boots
// 因为逆变的关系，animalPrinter不知道对应Cat类的CatPrinter的细节，只能按照animal的方式输出
printMyCat(animalPrinter, Cat("Boots")) // The animal's name is: Boots
```
## Invariance
Scala中的泛型类默认是不变类。即它们既不是covariant也不是contravariant。

# 18 上类型边界及下类型边界
在Scala中，type parameters及abstract type parameters可能会被限制在类型边界中。类型边界限制了type variables的具体值，也揭示了关于该类的成员信息。
## Upper Type Bounds
上类型边界**T <: A**声明了类型变量T是类型A的subtype，表示了类型T的上界（包含上界）。
```Scala
abstract class Animal {
  def name: String
}

abstract class Pet extends Animal {}

class Cat extends Pet {
  override def name: String = "Cat"
}

class Dog extends Pet {
  override def name: String = "Dog"
}

class Lion extends Animal {
  override def name: String = "Lion"
}

class PetContainer[P <: Pet](p: P) {
  def pet: P = p
}

val dogContainer = new PetContainer[Dog](new Dog)
val catContainer = new PetContainer[Cat](new Cat)

// 因为Lion不是Pet的子类，所以该句无法编译
val lionContainer = new PetContainer[Lion](new Lion)
```
## Lower Type Bounds
下类型边界**B >: A**声明了类型参数（或抽象类）B是类型A的supertype，表示了类型B的下界（包含上界）。大多数情况下，A将会是类的类型参数而B将会是一个方法的类型参数。
```Scala
// 错误例子，不能编译
// +B 代表Node与它的子类是Covariant关系
trait Node[+B] {
  def prepend(elem: B): Node[B]
}

case class ListNode[+B](h: B, t: Node[B]) extends Node[B] {
  // 参数elem类型为B，而B是协变的。这句是错误的，原因是：
  // 函数的参数类型中是逆变的，而这里结果的类型却是协变的
  def prepend(elem: B): ListNode[B] = ListNode(elem, this)
  def head: B = h // Node包含一个类型为B的成员head
  def tail: Node[B] = t // Node包含对剩余列表（tail）的引用
}

// Nil表示一个空列表
case class Nil[+B]() extends Node[B] {
  def prepend(elem: B): ListNode[B] = ListNode(elem, this)
}


// 为了解决上述错误，需要对参数elem的类型的协变特性进行翻转。
// 为此，引入一个以类型B作为下类型边界的新类型参数U
trait Node[+B] {
  def prepend[U >: B](elem: U): Node[U]
}

case class ListNode[+B](h: B, t: Node[B]) extends Node[B] {
  // U必须是B的父类，那么U将不必是协变的，这与函数的逆变特性相匹配
  def prepend[U :> B](elem: U): ListNode[U] = ListNode(elem, this)
  def head: B = h
  def tail: Node[B] = t
}

case class Nil[+B]() extends Node[B] {
  // U必须是B的父类，适用于U的prepend方法
  def prepend[U >: B](elem: U): ListNode[U] = ListNode(elem, this)
}
```
示例：
```Scala
trait Bird
case class BirdTypeA() extends Bird
case class BirdTypeB() extends Bird // covariant

val birdTypeAList = ListNode[BirdTypeA](BirdTypeA(), Nil())
val birdList: Node[Bird] = birdTypeAList
birdList.prepend(BirdTypeB())
```
# 19 内部类
在Scala中，可以让类拥有其他类作为成员。在类似java的语言中，这样的内部类是外围类的成员，而**在Scala中，这样的内部类被绑定到外部对象**。假设我们希望编译器在编译时避免混淆哪个节点属于哪个图，依赖路径的类型（path-dependent types）提供了一个解决方案。
```Scala
class Graph {
  class Node {
    var connectedNodes: List[Node] = Nil
    def connectTo(node: Node): Unit = {
      if (!connectedNodes.exists(node.equals)) {
        connectedNodes = node :: connectedNodes
      }
    }
  }
  var nodes: List[Node] = Nil
  def newNode: Node = {
    val res = new Node
    nodes = res :: nodes
    res
  }
}

val graph1: Graph = new Graph
val node1: graph1.Node = graph1.newNode
val node2: graph1.Node = graph1.newNode
node1.connectTo(node2) // 正确

val graph2: Graph = new Graph
val node3: graph2.Node = graph2.newNode
// 错误（ error: type mismatch），graph1.Node与graph2.Node不同，不同类型的Node无法连接
// 在Java中可以运行，因为Java中会将二者识别为同一类：Graph.Node
node1.connectTo(node3) 

//在Scala中要实现在Java中同样的效果，可以用Graph#Node：
class Graph {
  class Node {
    var connectedNodes: List[Graph#Node] = Nil
    def connectTo(node: Graph#Node): Unit = {
      if (!connectedNodes.exists(node.equals)) {
        connectedNodes = node :: connectedNodes
      }
    }
  }
  var nodes: List[Node] = Nil
  def newNode: Node = {
    val res = new Node
    nodes = res :: nodes
    res
  }
}
```
# 20 抽象类成员
抽象类（abstract type），比如trait及abstract class。可以有抽象类成员。这意味着为具体的实现定义了实际的类型。Trait或有着抽象类型成员的Calss经常与匿名类的实例化结合使用。
```Scala
// 定义一个Trait，其具有抽象类型type T，用于描述element的类型
trait Buffer {
  type T
  val element: T
}

// 在一个抽象类中扩展上述trait，为类型T添加上类型边界。这一抽象类通过声明T必须是
// Seq[U]的子类使得我们只能在该Buffer中存储sequence
abstract class SeqBuffer extends Buffer {
  type U
  type T <: Seq[U]
  def length = element.length
}

// 一个sequence buffer引用整数构成的List的例子
abstract class IntSeqBuffer extends SeqBuffer {
  type U = Int
}

def newIntSeqBuf(elem1: Int, elem2: Int): IntSeqBuffer = 
  /*
  newIntSeqBuf使用了IntSeqBuffer的匿名类实现（new IntSeqBuffer）来设置
  抽象类型T为具体类型List[Int]
  */
  new IntSeqBuffer {
    type T = List[U]
    val element = List(elem1, elem2)
  }
val buf = newIntSeqBuf(7, 8)
println("length = " + buf.length)
println("content = " + buf.element)
```

# 21 混合类型
一个对象的类型是多个类型的子类的情况在Scala中可以通过混合类型（多个类型的交集）来表示。混合类型可以有多个对象类型组成并包含一个refinement，该refinement可用于缩窄现有对象成员的签名（signature）。混合类型的一般形式为：A with B with C ... {refinement}
```Scala
// 可复制
trait Cloneable extends java.lang.Cloneable {
  override def clone(): Cloneable = {
    super.clone().asInstanceOf[Cloneable]
  }
}
// 可重置
trait Resetable {
  def reset: Unit
}

// 要实现一个函数cloneAndReset，以一个对象作为输入，复制该对象并将原始对象重置
def cloneAndReset(obj: Cloneable with Resetable): Cloneable = {
  val cloned = obj.clone()
  obj.reset
  cloned
}
```

# Self类型
Self类型（Self-type）是用于：声明一个trait必须被混入（mixed in）另一个trait的方法，即使它没有被直接扩展（extend）。这使得依赖的成员即使不导入也可以使用。  
Self-type是一种缩窄**this**类型的方法。为了在一个trait中使用self-type，需要写一个标识符，加上另一个需要混入的trait的类型，再加上一个**=>**：someIdentifier: SomeOtherTrait =>
```Scala
trait User {
  def username: String
}

trait Tweeter {
  // 下面这句的使用，使得无需导入便可以获取User中的username字段
  this: User =>
  def tweet(tweetText: String) = println(s"$username: $tweetText")
}

// 由于Tweeter中使用了self-type，所以扩展了Tweeter的VerifiedTweeter也必须扩展User
class VerifiedTweeter(val username_: String) extends Tweeter with User {
  def username = s"real $username_"
}

val someOne = new VerifiedTweeter("Rachael")
some.tweet("Hi there.")
```

# 22 隐式参数
一个方法可以有一个隐式参数（implicit parameter）列表，在参数列表的前面用**implicit**来标记。如果参数列表中的参数没有被正常传递，那么Scala将查看是否可以获得正确类型的隐式值，如果可以，该隐式值就会被自动传递。  
Scala寻找隐式参数值的地方有两个：  
 - 调用含有隐式参数的方法时，首先查找可以直接获取（无需前缀）的implicit定义及implicit参数
 - 然后在与implicit候选类型关联的所有伴生对象中查找被标记为implicit的成员

```Scala
abstract class Monoid[A] {
  def add(x: A, y: A): A // 定义add操作
  def unit: A // 定义基本单元，类型也为A
}

// 创建一个单例对象
object ImplicitTest {
  // 用于String，implicit关键字表明对应的对象可以被隐式地使用（directly）
  implicit val stringMonoid: Monoid[String] = new Monoid[String] {
    def add(x: String, y: String): String = x concat y
    def unit: String = ""
  }

  // 用于Int
  implicit val intMonoid: Monoid[Int] = new Monoid[Int] {
    def add(x: Int, y: Int): Int = x + y
    def unit: Int = 0
  }

  // 让参数m变得implicit，这样使得后续调用该方法时只需要提供xs参数，Scala可以找到对应的Monoid[A]
  def sum[A](xs: List[A])(implicit m: Monoid[A]): A = 
    if (xs.isEmpty) m.unit
    else m.add(xs.head, sum(xs.tail))
  
  def main(args: Array[String]): Unit = {
    println(sum(List(1, 2, 3))) // 隐式地使用intMonoid，输出6
    println(sum(List("a", "b", "c"))) // 隐式地使用stringMonoid，输出abc
  }
}
```

# 23 隐式转换
从类型S到类型T的隐式转换用一个具有函数类型**S => T**的隐式值（或者用一个可以转换为该类型的隐式方法）定义。隐式转换有两种应用场景：  
 - 如果表达式e的类型为S，并且S不满足期望类型T
  
    这种情况下，搜索适用于e且结果类型为类型T的转换
 - 在类型为S的e的选择e.m中，如果m不是S的成员  
  
    这种情况下，搜索适用于e并且结果包含成员m的转换

如果隐式方法**List[A] => Ordered[List[A]]**，以及隐式方法**Int => Ordered[Int]**存在，那么下述对两个类型为**List[Int]**的操作是允许的： 
```Scala
/* 
隐式方法Int => Ordered[Int]是通过scala.Predef.intWrapper（隐式导入）自动提供
scala.Predef对常用的类型（scala.collection.immutable.Map为Map）和方法（assert）
声明了一些别名，也声明了一些隐式转换
*/
List(1, 2, 3) <= List(4, 5)
```
一个隐式方法**List[A] => Ordered[List[A]]**的例子：
```Scala
import scala.language.implicitConversions

implicit def list2ordered[A](x: List[A])
    (implicit elem2ordered: A => Ordered[A]): Ordered[List[A]] = 
  new Ordered[List[A]] {
    def compare(that: List[A]): Int = 1
  }
```
因为如果不加区分地使用隐式转换，那么编译器会在编译隐式转换定义时发出警告。关闭警告可以采用如下任意两种方法之一：
 - 在隐式转换定义范围内导入scala.language.implicitConversions
 - 使用language:implicitConversions调用编译器

# 24 多态方法及类型推断
Scala中的方法既可以通过类型参数化，也可以通过值参数化。语法类似于泛型类。类型参数用方括号括起来，而值参数用圆括号括起来。并不总是需要显式地提供类型参数，编译器一般都可以基于上下文推断值变量的类型。
```Scala
def listOfDuplicates[A](x: A, length: Int): List[A] = {
  if (length < 1)
    Nil
  else
    x :: listOfDuplicates(x, length - 1)
}
// 由于通过[Int]显式地提供了类型参数，所以第一个变量必须是Int，返回的类型将是List[Int]
println(listOfDuplicates[Int](3, 4)) // List(3, 3, 3, 3)
// 编译器自己推断出值的类型是String
println(listOfDuplicates("La", 4)) // List(La, La, La, La)
```  
编译器一般情况下都可以自己推断值参数或返回结果的类型。但**对于递归方法，编译器无法推断结果类型**：
```Scala
// 如下例子将编译失败，因为没有指定返回类型
def fac(n: Int) = if (n == 0) 1 else n * fac(n - 1)
```
多态方法或泛型类实例化时也不强制指定类型参数，编译器将从上下文及实际方法/结构体参数中推断缺失的类型参数。**编译器从不推断方法（method）的参数类型**。然而，在某些情况下，**当函数作为变量传递时，它可以推断匿名函数的参数类型**。  
类型推断有时会推断出一个太过具体的类型：  
```Scala
val obj = null // 类型推断obj的类型为Null

// 这句将会无法编译，因为类型推断已经将obj推断为Null类型，所以无法再为其分配不同的值
obj = new AnyRef
```
为了可读性起见，应显式地指定类型。

# 25 算子
Scala中，算子（operator）本质是方法（method），任何有单个参数的方法可以被用作中缀运算符（infix operator）。如+可以用.来调用：10.+(1)，更可读的方式是写为中缀运算符：10 + 1  
## 定义及使用算子
```Scala
// + 示例
case class Vec(x: Double, y: Double) {
  def +(that: Vec) = Vec(this.x + that.x, this.y + that.y)
}

val vector1 = Vec(1.0, 1.0)
val vector2 = Vec(2.0, 2.0)

val vector3 = vector1 + vector2
vector3.x  // 3.0
vector3.y  // 3.0

// 逻辑示例
case class MyBool(x: Boolean) {
  def and(that: MyBool): MyBool = if (x) that else this
  def or(that: MyBool): MyBool = if (x) this else that
  def negate: MyBool = MyBool(!x)
}
def not(x: MyBool) = x.negate
def xor(x: MyBool, y: MyBool) = (x or y) and not(x and y)
```
## 算子优先级
当表达式使用多个算子时，算子是基于第一个字符来评估优先级。
```Scala
a + b ^? c ?^ d less a ==> b | c
// 上式等同于：
((a + b) ^? (c ?^ d)) less ((a ==> b) | c)
```
# 26 按名称参数与按值参数
按名称参数（by-name parameter）每次使用的时候都会评估一次，如果他们没有被使用则不会被评估（evaluated），这对于评估时需要大量计算或长时间运行某段代码的参数时可以帮助提高程序性能。对应的是按值参数（by-value parameter），好处是只需评估一次。为了使一个参数是by-name的，需要在其类型前面加上**=>**的前缀。  
```Scala
def calculate(input: => Int) = input * 37

// 一个循环的例子，该方法使用了两个参数列表来获取条件及循环体。如果条件为假，循环体就不会被评估
def whileLoop(condition: => Boolean)(body: => Unit): Unit = 
  if (condition) {
    body
    whileLoop(condition)(body)
  }

var i = 2
whileLoop(i > 0) {
  println(i)
  i -= 1
}
```
# 注释
注释（annotation）将元信息与定义关联起来。例如，方法前的注释@deprecated会导致编译器在使用该方法时打印警告。注释子句应用于它后面的第一个定义或声明。一个定义和声明之前可以有多个注释子句。这些子句的先后顺序不重要。
```Scala
object DeprecationDemo extends App {
  @deprecated("deprecation message", "release # which deprecates method")
  def hello = "aloha"

  hello
}
// 编译时会打印警告："“there was one deprecation warning(since release # which deprecates method)"
```
## 用注释确保编码的正确性
特定的注释会在条件不满足时造成编译失败。如@tailrec确保对应的方法是[尾递归](https://www.ruanyifeng.com/blog/2015/04/tail-call.html)（tail-recursive）的（tail-recursion可以保证程序的内存需求不变）。  
```Scala
import scala.annotation.tailrec

def factorial(x: Int): Int = {
  // factorialHelper必须满足尾递归才可以编译
  @tailrec
  def factorialHelper(x: Int, accumulator: Int): Int = {
    if (x == 1) accumulator else factorialHelper(x - 1, accumulator * x)
  }
  factorialHelper(x, 1)
}
```
## 注释影响代码生成
有些像@inline这样的注释会影响代码生成（即不使用的话jar文件可能会生成不同的字节码）。内联（inlining）意味着在方法体的调用点插入代码，因此产生的字节码会更长，但可能运行地更快。使用@inline不保证方法一定是内联的，但它会使得编译器尽可能尝试内联。  
当写和Java互操作的代码时，注释语法会有些不同。【确保对Java注释使用-target:jvm-1.8】。Java有着用户定义的形式为注释的元数据。注释的一个关键特性是它们依赖于指定name-value对来初始化它们的元素。如我们需要一个注释来跟踪某个类的Source，可以将其定义为：
```Scala
// 定义Source 
@interface Source {
  public String URL();
  public String mail();
}

// 对于实例化Java注释，需要使用命名参数（named arguments）
@Source(URL = "https://coders.com/",
        mail = "support@coders.com")
public class MyClass extends TheirClass ...

// 对于Scala
@Source(URL = "https://coders.com/",
        mail = "support@coders.com")
class MyScalaClass ...


// 如果注释只包含一个元素(没有默认值)，那么上述语法将非常繁琐
// 习惯上，如果一个名字被指定为value，它可以以类似于结构体的语法被用在Java中
@interface SourceURL {
  public String value();
  public String mail() default ""; // mail被制定了一个默认值，所以不需显式提供值
}

// 对于Java
@SourceURL("https://coders.com/")
public class MyClass extends TheirClass ...

// 对于Scala
@SourceURL("https://coders.com/")
class MyScalaClass ...
```

# 27 包及包的导入
## 创建包
通过在Scala文件的顶部声明一个或多个包名来创建包（package）。一种约定是将包命名为与包含Scala文件的目录相同的名称。然而，Scala与文件布局无关。还有一种声明package的方法是使用括号：
```Scala
// 这一声明方式包含了package的嵌套，并未scope和encapsulation提供了更好的控制
package users {
  package administrators {
    class NormalUser
  }
  package normalusers {
    class NormalUser
  }
}
```  
Package的命名习惯：**<top-level-domain>.<domain-name>.<project-name>**，如：
```Scala
package com.google.selfdrivingcar.camera

class Lens
```
## 导入包
```Scala
import users._  // import everything from the users package
import users.User  // import the class User
import users.{User, UserPreferences}  // Only imports selected members
import users.{UserPreferences => UPrefs}  // import and rename for convenience
```
Scala和Java的一点不同是：imports可以被用在任何地方：  
```Scala
def sqrtplus1(x: Int) = {
  import scala.math.sqrt
  sqrt(x) + 1.0
}

// 在发生命名冲突的情况下，你需要从项目的根导入一些东西，在包名前面加上_root_:
package accounts

import _root_.users._
```   
Scala中，java.lang和object Predef默认都会被自动导入。

## 包对象
Scala将包对象（package object）作为一个方便的容器在整个包中共享。包对象可以包含任意定义，而不仅仅是变量和方法定义。例如，它们经常用于保存包范围的类型别名和隐式转换。包对象甚至可以继承Scala类和特征。按照惯例，包对象的源代码通常放在名为package.scala的源文件中。每个包允许有一个包对象。放在包对象中的任何定义都被认为是包本身的成员。
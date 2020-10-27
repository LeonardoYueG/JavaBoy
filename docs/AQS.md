# AQS

## 一、AQS定义

AQS：AbstractQueuedSynchronizer(队列同步器)，用于构建锁和同步组件（类和容器）的框架。

**核心内容：**

1. 同步状态（volatile int state）的获取与修改（CAS实现）；
2. 内置FIFO同步队列实现线程的队列。

**AQS与锁的关系**

- AQS面向的是锁、同步容器实现者：简化了锁、同步的实现，屏蔽了同步状态的获取、同步队列管理、等待与唤醒的细节操作；
- 锁面向使用者：定义使用者与锁的交互接口，隐藏了同步的实现的细节。

## 二、AQS底层实现

同步队列 + 同步状态修改

### 1、同步队列

当线程获取同步状态失败，同步器会将线程以及等待的信息构成一个节点插入队列（双向FIFO）中，同步状态释放的时候，又从同步队列中唤醒节点。

**Node包含的信息**

- 线程的引用
- 在队列中的前驱后驱节点
- 等待状态

**细节**

- 同步器拥有head和tail节点分别指向同步队列的头尾节点（有head、首两个相似的概念，head节点指已经获取通过状态的节点，首节点是还在排序未获得同步状态的第一个节点）。
- 节点的插入是尾插法，需要使用到CAS，因为一个时候可能多个线程被阻塞，节点移除的时候不需要因为都是移除节点将自己的后继节点设置为首节点（不会冲突）。

### 2、同步状态的获取与释放

针对不同的使用场景，有三种不同的对同步状态的操作方式。

- 独占式：同一时刻**仅一个线程**能获取同步状态；
- 共享式：可**多个线程**同时获取/释放同步状态；
- 超时式：在**一定时间内**获取锁，如果超过一定时间，自动退出。

### 2.1、独占式

同步状态的获取与释放还分为**公平**和**非公平**，不同之处在于公平锁只有head节点才能获取同步状态。

**获取**

1. 获取同步状态 tryAcquire()；
2. 如果失败，生成同步节点 Node.EXCLUSIVE；
3. 将同步节点加入队列，addWaiter(Node.EXCLUSIVE) ，先直接CAS快速加入队尾，如果失败使用end(node)(也是CAS方式)一直自旋加入，让加入队尾的操作变成一个串行操作；
4. 在队列中**一直自旋**获取同步状态acquireQueued(Node node)；
5. 只有前驱节点是head节点的节点才能够获取同步状态if(pre == head && tryAcquire(args))。

**释放**

使用release()释放同步状态，并唤醒后继节点。

1. 先tryRelease()释放同步状态，如果成功执行2
2. 先判断是头节点和等待状态不为0（h != null && h.waitState != 0），再使用unparkSuccessor(Node)，的LockSupport（）唤醒等待后继节点

### 2.2、共享式

**获取**

1. 先直接tryAcquireShared(agrs) < 0,如果大于0直接获取到，如果小于0，与独占式一样生成节点放入同步队列中，并进入（2）的自旋获取状态
2. doAcquireShared()，始终自旋，先判断是否是head节点，如果是再使用r =tryAcquireShared(agrs) ，r > 0获取成功，退出自旋

**释放**

1. 可能有多个线程同时释放同步状态，需要使用CAS的方式进行释放

与独占式的主要区别：在同一时刻能否被多个线程同时获取同步状态

### 2.3、超时式

**获取**

和独占式类似，自旋获取时，不同地方在于如果没有获取成功，会判断是否超时

- 超时则退出
- 未超时重新计算休眠时间 = 设置的定时时间 - （当前时间 - 睡眠之前时间）

当等待时间小于1000纳秒时，不再睡眠（时间太短判断可能不准），直接自旋获取。

**释放**

​		和独占式类似

## 三、自定义同步组件的实现

一般自定义同步器的实现是基于模板的方法：在类的内部定义继承于AQS的内部类，内部类会继承所有AQS的模板方法，使用者可以根据自己需求重写部分类（有限的5个），重写的时候需要用到三个线程安全的获取、设置、修改同步状态的方法。

- 模板方法（很多）: 可以一个都不重写，直接调用；
- 可重写方法（5个）: 1.独占式获取\释放同步状态; 2.共享式获取\释放同步状态; 3.同步器当前是否被占用，这儿未包含超时式同步状态获取方式；
- 线程安全同步状态获取\修改方法：同样是模板方法，重写上述方法时需要，已经封装好了，不需要自己再重写，降低定义的难度，getState()、setState()、compareAndSetState()(CAS方式设置同步状态)。

## 四、基于AQS实现的组件

使用基于AQS实现的并发的组件

- 锁：可重入锁ReentrantLock、读写锁ReentrantReadWriteLock；
- 同步器：LockSupport工具、condition接口;
- 容器：ConcurrentHashMap、队列（阻塞和非阻塞）;
- 并发类（框架）：Fork/Join框架、原子类、并发工具类.

### 1.锁

#### 1.1、ReentrantLock

同一个线程可多次获取，但是是互斥锁，不可多个线程同时获取。

特点：可重入、可设置是否公平、互斥

##### 可重入

任意线程获取锁之后能再次获取该锁而不被阻塞（同步状态 == 0，未被获取，> 0已被获取）

实现

1. 线程再次获取：state == 0时，未被获取，直接CAS获取，state > 0 时，判断当前线程是否占有锁的线程，是则获取成功
2. 锁释放: 获取多少次，就必须释放多少次，当state == 0时，才时释放成功

##### 公平与非公平

之前讲的是公平，每一个锁都在自旋，只有头节点才获取到同步状态，公平锁和非公平锁唯一不同的地方在于在判断条件新增了判断是否有前驱节点。

公平锁实现：当前节点是否是有前驱节点（没有就是头节点，才能获取）

公平锁问题：效率低

##### 互斥锁

最多一个线程访问

#### 1.2、ReentrantReadWriteLock

读写分离，并发度提高，分为两个锁：读锁readLock() + 写锁 writeLock()

相较于重入锁新增的特点：

- 双锁结构
- 共享锁
- 锁降级：将写锁降为读锁

**实现**

- 读写状态的维护
- 写锁的获取与释放
- 读锁的获取与释放

**读写状态的维护**

同样使用一个同步状态，只不过一个同步状态维护了两种状态：高16位表示读，低16位表示写，先通过位运算再获取修改读写状态

**写锁**

- 获取

  - 如果本线程已经获取了写锁：直接增加写锁同步状态
  - 如果其他线程获取了读写状态，则等待，不管是不是自己线程获取了读状态都要等待，原因：因为要保证写锁的状态对读锁可见，如果在读锁的状态下获取写锁，那么其他读线程就不知道写锁的操作，所以必须读锁全部释放完才能写锁

- 释放

  - 与重入锁类似

**读锁**

- 获取

  - 只要其他线程没有获取写锁，直接CAS获取读状态，增加读同步状态，否在等待。
  - 细节：每个线程的自己的读状态保存在ThreadLocal中

- 释放

  - 与重入锁类似

**锁降级**

- 当前线程获取写锁之后，同时再获取读锁，再释放写锁的过程，目的是数据的可见性（保证当前线程能感知到数据的变化）

  - 当前线程如果直接释放写锁，再获取读锁，可能其他线程会先获取写锁，而本线程获取不到读锁，从而感知不到数据的更新

### 2、同步器

#### 2.1、LockSupport工具

一组公共静态方法，提供基础的线程阻塞LockSupport.park()和唤醒功能LockSupport.unpark()，提供限时阻塞功能。

#### 2.2、condition接口

```java
Lock lock = new ReentrankLock();       
Condition condition = lock.newCondition();
condition.await();
condition.signal();
```

**与Object监视器方法的区别（P147）**

|    对比项    | Object监视器方法 |  Condition   |
| :----------: | :--------------: | :----------: |
|     类型     |   基于对象的锁   | 基于Lock对象 |
| 等待队列数量 |      仅一个      |     多个     |
|  是否可中断  |      不支持      |     支持     |
|   限时获取   |      不支持      |     支持     |

**Condition实现**

condition也是AQS的子类，因为conditon操作也需要获取同步状态。

**等待队列**

- 每个condition对象都拥有一个等待队列；
- 添加到等待队列中不需要CAS操作，因为调用await()方法说明已经获取到锁；
- 同步队列是满足条件，但是没有获取到同步状态；
- 等待队列是获取了同步状态，但未满足某种条件而进入等待队列。

等待/通知

- 等待await()

  调用await()方法，生成Node节点，线程释放锁，并变为等待状态和进入等待队列。

  - 1.将当前线程构造为节点加入等待队列（不是同步队列中的原节点，会构造新的等待队列节点，参数不一样）
  - 2.释放同步状态
  - 3.唤醒在同步队列中的后继节点
  - 4.线程进入等待状态

- 通知signal()

  - 1.调用该方法线程需要已经获取锁
- signalAll()相当于把所有等待队列中节点移动到同步队列，唤醒每一个节点；
    - signal()唤醒等待队列中head节点

  - 2.将等待队列head节点通过enq(Node)方式移动到同步队列（自旋CAS添加），并唤醒该节点对应线程（Locksupport(）的方式唤醒）

  - 3.和AQS其他同步器一样，开始自旋获取同步状态（锁）

### 3、容器

#### 3.1、ConcurrentHashMap

详情间Java容器内容

1. 使用ConcurrentHashMap的原因

2. 结构

3. 操作

   - get()

     - 两次hash定位到segment，再hash定位到具体位置
     - get()高效之处：get过程中不需要加锁，而hashtable需要，因为get()时候的共享变量被定义为volatile变量，其他线程是可见的
     - 定位segment和hashEntry的hash算法一样，但是和length-1 &的值不一样

   - put()

     需要加锁

     流程：1.判断segment是否需要扩容，2.定位添加元素的位置，添加元素

     先判断是否需要扩容，可以避免无效扩容（hashmap这样扩容），只扩容segment，不扩容整个数组

   - size()

     - 先不加锁（加锁的成本太高了），判断两次结果是否相同，如果连续三次都不相同，加锁。

#### 3.2、队列

同步队列包含阻塞队列和非阻塞队列

- 非阻塞队列基于CAS实现；
- 阻塞队列基于锁的通知模式实现。

##### 3.2.1、非阻塞队列

基于CAS实现，常见ConcurrentLinkedQueue，ConcurrentLinkedQueue无界线程安全队列，head 和tail节点，其他节点通过next相连接。

**tail节点和尾节点**

- 尾节点：队列最后一个节点；
- tail节点：tail指针指向的节点；

**实现**

- **入队**：定位尾节点、加入节点、更新tail节点

  - 定位尾节点（tail节点或者tail节点的next节点）
- 通过CAS将入队节点设置为当前尾节点的next节点
  - 更新tail节点
- 1.如果tail.next != null, 则将入队节点设置为tail节点
    - 2.如果tail.next == null，则将入队节点设置为tail.next节点

  原因：提高效率，如果tail就是尾节点，每次都CAS的方式去更新tail，效率比较低，上述方法可以减少对tail的更新，从而提高并发度。

- **出队**：弹出head节点所指向的节点

  - 直接弹出head节点，不是每次都要更新head节点，不为空直接CAS弹出数据，只有head里面没有数据才会更新head节点，减少使用CAS更新head的节点消耗（和tail一样）

##### 3.2.2、阻塞队列

阻塞队列（BlockingQueue）是一个支持两个附加操作（阻塞插入put(e)、阻塞移除take()）的队列。

- 支持阻塞的插入方法：当队列满时，队列会阻塞插入元素的线程，直到队列不满；
- 支持阻塞的移除方法：在队列为空时，获取元素的线程会等待队列变为非空。

**实现**

采用通知模式实现（需要用到Lock锁），当生产者往满队列里面添加元素就会阻塞（condition.await()），当消费者消费元素就会通知生产者可用condition.signal()，同理消费者消费空队列就会阻塞，当生产者生产了就会通知消费者。

**容器**

- ArrayBlockingQueue：数组有界，默认非公平访问
- LinkedBlockingQueue：链表有界，但是默认和最大长度都是Integer.MAX_VALUE

- PriorityBlockingQueue：优先级无界
- DelayQueue：支持延时获取元素的无界，需要实现Delay接口，指定元素被获取的时间，延时期满才能获取元素。运用场景：1.缓存系统：DelayQueue保存元素的缓存期时间，到达缓存期则说明元素过期。2.定时调度任务，DelayQueue保存任务执行的时间，到达定时时间则执行
- SynchronousQueue：不保存元素的阻塞队列，默认非公平，效率高，适合需要传递元素的场景
- LinkedTransferQueue：链表无界，多了transfer()和tryTransfer()方法，transfer()方法尝试直接将元素交给消费者，如果没有则阻塞，tryTransfer()尝试交给消费者，没有消费者立即返回false，而transfer()会阻塞。
- LinkedBlockingDeque：链表双端，双端插入移除减少竞争。

### 4、类（框架）

#### 4.1、Fork/Join框架

主要用于并行计算中，把大的计算任务拆分成多个小任务并行计算。

- fork()：任务分割
- join()：结果合并

**ForkJoinPool** 

ForkJoin 使用 ForkJoinPool 来启动，它是一个特殊的线程池，线程数量取决于 CPU 核数。

ForkJoinPool 由 ForkJoinTask 数组和 ForkJoinWorkerThread 数组组成。

- ForkJoinTask 数组：存放程序提供的 ForkJoinTask；
- ForkJoinWorkerThread 数组：负责执行这些任务。

ForkJoin 在执行时需要时 ForkJoin 任务，但是通常我们使用 ForkJoinTask 的子类，ForkJoinTask 任务和普通任务的主要区别在于需要实现 compute() 方法，在compute()方法中如果任务小于阈值则直接执行，否则使用fork()和join()分解任务合并结果，类似于一个递归操作。

ForkJoinTask 的子类

- RecursiveAction：用于**没有**返回结果的任务
- RecursiveTask：用于**有**返回结果的任务

**优缺点：**

**优点**：ForkJoinPool 实现了工作窃取算法来提高 CPU 的利用率。每个线程都维护了一个双端队列，用来存储需要执行的任务。当一个线程的任务执行完成之后，会从其他线程执行的双端队列中“窃取任务”，为了降低竞争，会从队列的尾部进行任务窃取。

**缺点**：创建队列和多线程会消耗资源，同时队列中只有一个任务的时候也会出现竞争。

#### 4.2、原子类

操作不可中断的类型，实现方法是利用 CAS (Compare And Swap) + volatile 和 native 方法来保证原子操作，从而避免 synchronized 的高开销，执行效率大为提升。

- 基础类型
- 数组类型
- 应用类型
- 对象的属性修改类型

##### 2.1、基础类型

- AtomicInteger:原子更新整型

  **具体方法**

  ```java
  public final int get() //获取当前的值
  public final int addAndGet(int value)//以原子方式将输入值和实例中的值相加并返回相加后的值
  public final int getAndSet(int newValue)//获取当前的值，并设置新的值
  public final int getAndIncrement()//获取当前的值，并自增
  public final int getAndDecrement() //获取当前的值，并自减
  public final int getAndAdd(int delta) //获取当前的值，并加上预期的值
  public final void lazySet(int newValue)//最终会变为设置值，但是中途一段时间为原值
  boolean compareAndSet(int expect, int update) //CAS设置元素
  ```

  **实现**

  - 使用native的CAS方法，Unsafe只有三种CAS方法，其他方法要实现原子更新类型都是通过封装下面三种方法实现的，下面两种基础类型和本方法类似。

    - 1.compareAndSwapObject
    - 2.compareAndSwapInteger
    - 3.compareAndSwapLong

- 2.AtomicBoolean

- 3.AtomicLong

##### 2.2、数组类型

- AtomicIntegerArray：具体方法：addAndGet(int index, int value)更新特定位置的值

- AtomicReferenceArray
- AtomicLongArray

##### 2.3、引用类型

- AtomicReference：引用类型原子类
- AtomicReferenceFieldUpdater:原子更新引用类型里的字段
- tomicMarkableReference：更新带标记位的引用类型

##### 2.4、对象的属性修改

- AtomicIntegerFieldUpdater：更新整型的字段的更新器
- AtomicLongFieldUpdater：原子更新长整形字段的更新器
- AtomicStampedReference:更新带有版本号的引用类型，将数值和引用关系起来，可原子更新数组和引用，解决CAS的ABA问题

#### 4.3、并发工具类

##### 3.1、并发控制

用于控制线程关系同步

- CountDownLatch
- CyclicBarrier
- Semaphore

###### 3.1.1、CountDownLatch

用于控制一个线程等待多个线程，维护了一个计数器 cnt，每次调用 countDown() 方法会让计数器的值减 1，减到 0 的时候，那些因为调用 await() 方法而在等待的线程就会被唤醒。

###### 3.1.2、CyclicBarrier

与CountDownLatch相似，用于控制线程之间的相互等待，只有线程都达到指定位置的时候才继续执行。

和CountDownLatch不同之处在于：

- 有reset() 有重置计数器，可多次使用，CountDownLatch类似于一次性的；
- CyclicBarrier 有两个构造函数，其中CyclicBarrier(int parties, Runnable barrierAction)， parties 指示计数器的初始值，barrierAction 在所有线程都到达屏障的时候会执行一次。

###### 3.1.3、Semaphore

Semaphore类似于操作系统中信号量，可以控制对互斥资源的访问线程数。

##### 3.2、线程数据交换

Exchanger


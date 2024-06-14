#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/signal.h>
#include <linux/sched/task.h>

MODULE_LICENSE("GPL");

struct wait_opts{
	enum pid_type wo_type;
	int wo_flags;
	struct pid * wo_pid;
	struct waitid_info * wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};


extern pid_t kernel_clone(struct kernel_clone_args *args);
extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,const char __user *const __user *__argv, const char __user *const __user *__envp);
extern struct filename * getname_kernel(const char __user * filename);


static struct task_struct *task;
int status;

int my_exec(void *argc){//child process executes the test program
	const char path[] = "/home/vagrant/csc3150/Assignment1/program2/test.c";

	struct filename *FN = getname_kernel(path);

	printk("[program2] : child process\n");
	int num=do_execve(FN,NULL,NULL);

	if(!num){
		return 0;
	}else{
		do_exit(num);
	}
	return 0;
}

void parent_wait(pid_t pid){
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type ptype;
	ptype=PIDTYPE_PID;
	wo_pid=find_get_pid(pid);

	wo.wo_type=ptype;
	wo.wo_pid=wo_pid;
	wo.wo_flags=WEXITED;
	wo.wo_info=NULL;
	wo.wo_stat=&status;
	wo.wo_rusage=NULL;

	int a;

	a=do_wait(&wo);
	
	put_pid(wo_pid);
	return;
}

//implement fork function
int my_fork(void *argc){
	
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using kernel_clone or kernel_thread */
	pid_t pid;
	struct kernel_clone_args solve={
		.flags=SIGCHLD,
		.stack=(unsigned long)&my_exec,
		.stack_size=0,
		.parent_tid=NULL,
		.child_tid=NULL,
		.tls=0
	};
	pid=kernel_clone(&solve);

	/* execute a test program in child process */
	//Print out the process id for both parent and child process.(5) 
    printk("[program2] : The child process has pid = %d\n", pid);
    printk("[program2] : The parent process has pid = %d\n", (int) current->pid);
	
	/* wait until child process terminates */
	parent_wait(pid);

	return 0;
}

static int __init program2_init(void){

	printk("[program2] : Module_init {Yang Yin} {120090516}\n");
	
	/* write your code here */
	
	/* create a kernel thread to run my_fork */
	printk("[program2] : module_init create kthread start\n");
	task=kthread_create(&my_fork,NULL,"MyThread");

	if(!IS_ERR(task)){
        printk("[program2] : module_init kthread start");
        wake_up_process(task);
    }

	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);

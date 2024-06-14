#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[])
{

	int status;
	pid_t pid;
	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();

	if (pid < 0)
	{
		perror("fork");
		exit(1);
	}

	else
	{
		if (pid == 0)
		{
			char *arg[argc];

			for (int i = 0; i < argc - 1; i++)
			{
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			printf("I'm the Child Process, my pid = %d\n", getpid());

			/* execute test program */
			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);

			printf("Continue to run original child process!\n");
			perror("execve");
			// exit(EXIT_FAILURE);
			raise(SIGCHLD);
			exit(0);
		}
		else
		{
			// sleep(3);
			printf("I'm the Parent Process, my pid = %d\n", getpid());
			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			// wait(&status);
			/* check child process'  termination status */
			printf("Parent process receives SIGCHLD signal\n");
			if (WIFEXITED(status))
			{ // normal exit
				printf("Normal termination with EXIT STATUS = %d\n",
					   WEXITSTATUS(status));
			}
			else if (WIFSIGNALED(status))
			{ // abnormal exit
				int AE = WTERMSIG(status);
				printf("CHILD EXEUTION FAILED: %d\n", AE);
				// switch (AE)
				// {
				// case 6:
				// 	printf("Child process receives SIGABRT signal\n");
				// 	break;
				// case 7:
				// 	printf("Child process receives SIGBUS signal\n");
				// 	break;
				// case 8:
				// 	printf("Child process receives SIGFPE signal\n");
				// 	break;
				// case 14:
				// 	printf("Child process receives SIGALRM signal\n");
				// 	break;
				// default:
				// 	break;
				// }
			}
			else if (WIFSTOPPED(status))
			{
				printf("CHILD PROCESS STOPPED\n");
			}
			else
			{
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}

	return 0;
}

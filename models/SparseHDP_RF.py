import torch
from torch.optim import Adam
import numpy as np
np.random.seed(1)

class SparseTensorHDP:
    #m: #pseudo inputs
    #ind: observed entry indices
    #y: observed entry values
    #nvec: dim of each mode
    #B: batch-size
    #R1: number of socialbilities
    #R2: number of inherent factors
    def __init__(self, ind, y, nvec, R1, R2, m, B, device):
        self.device = device
        self.ind = ind
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.nvec = nvec
        self.R1 = R1
        self.R2 = R2
        self.m = m
        self.B = B
        self.nmod = len(nvec)

        #alpha parameter
        self.log_alpha = torch.tensor(np.log(1.0), device=self.device, requires_grad=False)
        #beta parameters, beta = softmax(beta_hat), top-level DP
        self.beta_hat = [torch.tensor(np.random.rand(self.nvec[k]+1), device=self.device, requires_grad=True) for k in range(self.nmod)]
        #omega paras, = softmax(omega_hat), socialbilities, (d_k + 1) \times R_1 in each mode
        self.omega_hat = [torch.tensor(np.random.rand(self.nvec[k]+1, self.R1), device=self.device, requires_grad=True) for k in range(self.nmod)]
        #gamma_r^k: K \times R_1, concentration parameter for each 2nd-level DP
        self.log_gamma = torch.tensor(np.zeros([self.nmod, self.R1]), device=self.device, requires_grad=True)
        #locations/atoms in each mode
        self.theta = [torch.tensor(np.random.rand(self.nvec[k], self.R2), device=self.device, requires_grad=True) for k in range(self.nmod)]
        self.R = self.R1 + self.R2
        #dim. of frequencies
        self.d = self.nmod*self.R
        #init mu, L, Z
        self.Z = torch.tensor(np.random.rand(self.m, self.d), device=self.device, requires_grad=True)
        self.N = y.size
        #variational posterior for the weight vector
        self.mu = torch.tensor(np.zeros([2*m,1]), device=self.device, requires_grad=True)
        self.L = torch.tensor(np.eye(2*m), device=self.device, requires_grad=True)
        #noise precision
        self.log_tau = torch.tensor(1., device=self.device, requires_grad=True)


    def cumsum_exclusive(self, a):
        ac = torch.cumsum(a[:-1], dim = 0)
        res = torch.cat([torch.tensor(np.array([0.0]), device=self.device), ac], 0)
        return res

    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        #recover beta, d_k+1 dim vector in each mode k
        beta = [torch.softmax(self.beta_hat[k], dim=0) for k in range(self.nmod)]
        #(d_k+1) \times R_1 in each mode k
        log_omega = [torch.log_softmax(self.omega_hat[k],dim=0) for k in range(self.nmod)]
        #scalar
        alpha = torch.exp(self.log_alpha)
        #K \times R_1 across all the modes
        gamma = torch.exp(self.log_gamma)

        log_GEM_beta = 0.0
        for k in range(self.nmod):
            #cum_sum without term corresponding to beta_dk
            cumbeta = self.cumsum_exclusive(beta[k][:-1])
            sum_log_lambda = torch.sum(torch.log(torch.tensor([1.0]) - cumbeta),0)
            log_GEM_beta = log_GEM_beta + self.nvec[k]*self.log_alpha + (alpha-1.0) * beta[k][-1] - sum_log_lambda

        log_Gamma_gamma = self.nmod*self.R1*self.log_alpha - alpha*torch.sum(gamma)

        log_Dir_omega = torch.sum(torch.lgamma(gamma))
        for k in range(self.nmod):
            #(d_k + 1) \times R_1
            Dir_params = torch.outer(beta[k], gamma[k,:])
            log_Dir_omega = log_Dir_omega - torch.sum(torch.lgamma(Dir_params)) + torch.sum((Dir_params - 1.0)*log_omega[k])

        #sum(K \times B \times R_1, 0) --> B \times R_1
        s = torch.sum(torch.stack([log_omega[k][self.ind[sub_ind,k],:] for k in range(self.nmod)]), 0)
        log_edge_prob = torch.sum(torch.logsumexp(s, dim = 1))  - np.log(self.R1) #constant

        #B \times K*R_1
        U_omega = torch.cat([self.omega_hat[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        U_theta = torch.cat([torch.sigmoid(self.theta[k][self.ind[sub_ind, k],:]) for k in range(self.nmod)], 1)
        #B \times K*R
        input_emb = torch.cat([U_omega, U_theta], 1)
        y_sub = self.y[sub_ind]
        #random Fourier feature mapping
        Phi = torch.matmul(input_emb, self.Z.T)
        Phi = torch.cat([torch.cos(Phi), torch.sin(Phi)], 1) #B \times 2m

        tau = torch.exp(self.log_tau)
        Ltril = torch.tril(self.L)
        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = log_GEM_beta + log_Gamma_gamma + log_Dir_omega \
               - 0.5*torch.sum(self.Z*self.Z) \
               - 0.5*self.m*torch.trace(hh_expt) \
               + 0.5*self.N*self.log_tau \
               - 0.5*tau*self.N/self.B*( torch.sum(torch.square(y_sub - torch.matmul(Phi, self.mu))) \
                                         + torch.sum(torch.square(torch.matmul(Phi, self.L))) ) \
               + self.N/self.B*log_edge_prob \
               + 0.5*torch.sum(torch.log(torch.square(torch.diagonal(Ltril))))

        return -torch.squeeze(ELBO)


    def pred(self, test_ind):
        #(d_k+1) \times R_1 in each mode k
        U_omega = torch.cat([self.omega_hat[k][test_ind[:, k],:] for k in range(self.nmod)], 1)
        U_theta = torch.cat([torch.sigmoid(self.theta[k][test_ind[:, k],:]) for k in range(self.nmod)], 1)
        inputs = torch.cat([U_omega, U_theta], 1)
        inputs = self.bn(inputs)
        Phi = torch.matmul(inputs, self.Z.T)
        Phi = torch.cat([torch.cos(Phi), torch.sin(Phi)], 1) #B \times 2m
        pred_mean = torch.matmul(Phi, self.mu)
        return pred_mean

    def _callback(self, ind_te, yte):
        with torch.no_grad():
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te)
            pred_tr = self.pred(self.ind)
            err_tr = torch.mean(torch.square(pred_tr - self.y))
            err_te = torch.mean(torch.square(pred_mean - yte))
            err_mae = torch.mean(torch.abs(pred_mean - yte))
            print('tau=%.5f, train_err = %.5f, test_err=%.5f, test_mae = %.5f' % \
                  (tau, err_tr, err_te, err_mae))
            with open('RF_NEST_HDP.txt','a') as f:
                f.write('%g '%err_te)

    def save_embeddings(self,file_name):
        with torch.no_grad():
            print(self.omega_hat[0].shape)
            print(self.theta[0].shape)

            embeddings = [torch.cat([self.omega_hat[k][:-1,:],torch.sigmoid(self.theta[k])],1) for k in range(self.nmod)]
            for k in range(self.nmod):
                np.save(file_name+"_"+str(k)+".npy",embeddings[k].numpy())



    def train(self, ind_te, yte, lr, max_epochs=100):
        yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        paras = self.beta_hat + self.omega_hat + [self.log_gamma] + [self.Z, self.mu, self.L, self.log_tau]

        minimizer = Adam(paras, lr=lr)
        for epoch in range(max_epochs):
            self.bn.train()
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                loss.backward()
                minimizer.step()
                curr = curr + self.B
            print('epoch %d done'%epoch)
            if epoch%5 == 0:
                self._callback(ind_te, yte)
                '''
                with torch.no_grad():
                    print('gamma')
                    gamma = torch.exp(self.log_gamma)
                    print(gamma)
                    for k in range(self.nmod):
                        print('mode %d'%k)
                        print('omega')
                        omega_k = torch.softmax(self.omega_hat[k], dim = 0)
                        print(omega_k)
                        print('beta')
                        beta_k = torch.softmax(self.beta_hat[k], dim = 0)
                        print(beta_k)
                '''


        print(self.mu)
        print(self.L)
        with torch.no_grad():
            print('gamma')
            gamma = torch.exp(self.log_gamma)
            print(gamma)
            for k in range(self.nmod):
                print('mode %d'%k)
                print('omega')
                omega_k = torch.softmax(self.omega_hat[k], dim = 0)
                print(omega_k)
                print('beta')
                beta_k = torch.softmax(self.beta_hat[k], dim = 0)
                print(beta_k)

        self._callback(ind_te, yte)

def test_alog():
    nfold = 5
    m = 100
    mse = []
    mse_train = []
    batch_size = 100
    lr = 0.001
    nepoch = 400
    for k in range(nfold):
        U = [np.random.rand(200,R), np.random.rand(100,R), np.random.rand(200,R)]
        ind = []
        y = []
        with open('../alog-pure/train-fold-%d.txt'%(k+1), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                y.append(float(items[-1]))
                ind.append([int(idx)-1 for idx in items[0:-1]])
            ind = np.array(ind)
            y = np.array(y)

        ind_test = []
        y_test = []
        with open('../alog-pure/test-fold-%d.txt'%(k+1), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                y_test.append(float(items[-1]))
                ind_test.append([int(idx)-1 for idx in items[0:-1]])
            ind_test = np.array(ind_test)
            y_test = np.array(y_test)
        R1 = 7
        R2 = 8
        nepoch = nepoch
        model = SparseTensorHDP(ind, y, [200, 100, 200], R1, R2, m, batch_size, torch.device('cpu'))
        model.train(ind_test, y_test, lr, nepoch)
        #model.save_embeddings("alog_embeds")


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    test_alog()
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.ndimage
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class qs_strikeslip_fault():
    def __init__(self, zdepth=3, xlength=3, Nz=10, Nx=10, G=30000., rho=2.5e-3, zeta=0., Ks_path="./Data/",
                 gamma_s=25., gamma_w=10., sigma_ref=100., depth_ini=0.e-6,
                 vinf=1.e-10, Dmu_estimate=.5):
        self.zdepth = zdepth
        self.xlength = xlength
        self.G = G  # [MPa]
        self.rho = rho  # [g/mm^3]
        self.zeta = zeta  # [-]
        self.gamma_s = gamma_s  # [MPa/mm]
        self.gamma_w = gamma_w  # [MPa/mm]
        self.sigma_ref = sigma_ref  # [MPa]
        self.depth_ini = depth_ini  # [mm]
        self.vinf = vinf
        self.Nx = Nx
        self.Nz = Nz
        self.N = Nx * Nz

        Dz = zdepth / Nz  # [km]
        self.Dz = Dz
        Dx = xlength / Nx  # [km]
        pts = np.arange(Nx * Nz)  # [-]
        self.pts_x = (pts % Nx) * Dx + Dx / 2.  # [km]
        self.pts_z = (pts // Nx) * Dz + Dz / 2.  # [km]
        self.coords = np.array([self.pts_x, self.pts_z]
                               ).swapaxes(0, 1)  # [km,km]

        infile = Ks_path + "kijs_" + str(Nx) + "x" + str(Nz) + "elem" + "-" + str(
            int(xlength)) + "x" + str(int(zdepth)) + "length" + ".npy"
        print("Loading:", infile)
        source_kijs = np.load(infile)  # [1/km]
        kijs = -source_kijs * 1e-6  # [1/mm]
        self.kijs = kijs

        q = zdepth / xlength  # [-]
        ref_l = 1.
        ref_mu = 1.
        nu = .5 * ref_l / (ref_l + ref_mu)
        ktheory = 2. / (np.pi * zdepth * 1.e6) * \
            (1. + q**2 / (1. - nu)) / (1. + q**2)**.5
        print("Theoretical value of total stiffness assuming a patch of dimensions Lx x Lz: ", ktheory)
        print("Approximate value of total stiffness assuming a patch of length Lx: ",
              1. / xlength*1e-6)
        print("Calculated value: ", np.sum(kijs) / (Nx * Nz))

        self._time_units()
        self._set_sigman_eff()
        args = [xlength, kijs, Dmu_estimate, self.sigma_eff, G, rho]
        self._set_scales(args)
        self._set_matrices(args)

    def _time_units(self):
        print("Units are in: mm,N,ms,MPa,gr")
        self.second = 1000.
        self.minute = 60 * self.second
        self.hour = 60 * self.minute
        self.day = 24 * self.hour
        self.week = 7 * self.day
        self.year = 52 * self.week

    def _set_scales(self, args):
        xlength, kijs, Dmu_estimate, sigma_eff, G, rho = args

        Ly = xlength * 1e6  # [mm] # in the direction of slip
        self.vs = np.sqrt(G / rho)  # [mm/ms]
        self.wp = self.vs / Ly  # [1/ms]
        self.Tscale = (1. / self.wp)  # [ms]
        self.Dscale = 1000. * 1e-4  # [mm]
        self.Pscale = self.G * self.Dscale / Ly  # [MPa]
        self.Vscale = self.Dscale / self.Tscale

    def _set_matrices(self, args):
        xlength, kijs, Dmu_estimate, sigma_eff, G, rho = args

        Ly = xlength * 1e6  # [mm] # in the direction of slip
        avgDmax = np.average(np.linalg.solve(kijs, -Dmu_estimate * sigma_eff / G))  # [mm]
        desDmax = np.average(-Dmu_estimate * sigma_eff * Ly / G)  # [mm]

        factor = avgDmax / desDmax
        Ks = kijs * factor  # [1/mm]
        self.Ks = Ks
        Elastic = G * self.Tscale**2 / (rho * Ly**2)  # [-]
        Damp = 2. * self.zeta * self.wp * self.Tscale  # [-]

        self.nvinf = self.vinf / self.Vscale  # [-]
        self.nKs = (Elastic * Ly) * Ks  # [-]
        self.nEs = Damp * sc.linalg.sqrtm(Ly * Ks)  # [-]
        self.nMs = np.eye(self.nKs.shape[0])
        self.nsigma_eff = sigma_eff / self.Pscale  # [-]

    def get_plant(self, normalized=True):
        if normalized:
            return self.nMs, self.nKs, self.nEs, self.nsigma_eff, self.nvinf
        else:
            return self.nMs * self.Pscale * self.Tscale**2 / self.Dscale, self.nKs * self.Pscale / self.Dscale, self.nEs * self.Tscale / self.Dscale, self.nsigma_eff * self.Pscale, self.nvinf * self.Vscale

    def _set_sigman_eff(self):
        sigma_n = self.gamma_s * \
            self.coords[:, 1] + self.gamma_s * self.depth_ini  # [MPa]
        pw = np.maximum(self.gamma_w * self.coords[:, 1] + self.gamma_w * self.depth_ini,
                        sigma_n - self.sigma_ref * np.ones(sigma_n.shape[0]))  # [MPa]
        self.sigma_eff = sigma_n - pw  # [MPa]

    def get_nominal(self):
        # Nominal quantities
        G0 = self.G * 1.1  # [MPa]
        rho0 = self.rho * 1.1  # [g/mm^3]
        Ly = self.xlength  * 1e6
        Ly0 = Ly * 1.1  # [mm]
        zeta0 = self.zeta * 0.8  # [-]

        Kijs0 = self.Ks * Ly / Ly0  # [1/mm]

        vs0 = np.sqrt(G0 / rho0)  # [mm/ms]
        wp0 = vs0 / Ly0  # [1/ms]

        N_hat0 = self.Pscale * self.Tscale**2 / \
            (rho0 * Ly0 * self.Dscale)  # [-]
        Elastic0 = G0 * self.Tscale**2 / (rho0 * Ly0**2)  # [-]
        Damp0 = 2. * zeta0 * wp0 * self.Tscale  # [-]

        k_hat0 = (Elastic0 * Ly0) * Kijs0  # [-]
        eta_hat0 = Damp0 * np.round(sc.linalg.sqrtm(Ly0 * Kijs0), 14)  # [-]

        nSigma_eff0 = np.mean(self.nsigma_eff)

        return N_hat0, k_hat0, eta_hat0, nSigma_eff0

    def get_coordinates(self):
        return self.pts_x, self.pts_z, self.Dz

    def plot_avgs(self, ts, ys, delta_zero=False):
        N = self.N
        if delta_zero:
            offset = -ys[0, :5 * N]
            ys[:, :5 * N] += offset

        avg_v = np.mean(ys[:, :N], axis=1) * self.Vscale
        avg_z = np.mean(ys[:, N:2 * N], axis=1) * self.Vscale
        avg_s = np.mean(ys[:, 2 * N:3 * N], axis=1) * self.Dscale
        avg_d = np.mean(ys[:, 3 * N:4 * N], axis=1) * self.Dscale
        avg_a = np.mean(ys[:, 4 * N:5 * N], axis=1) * self.Vscale / self.Tscale
        avg_p = np.mean(ys[:, 5 * N:6 * N], axis=1) * self.Pscale
        plt.plot(ts * self.Tscale / self.day, avg_v, "-o")
        plt.plot(ts * self.Tscale / self.day, avg_z)
        plt.ylabel("average velocity and slip rate [m/s]")
        plt.xlabel("time [day]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, avg_d)
        plt.plot(ts * self.Tscale / self.day, avg_s)
        plt.ylabel("avg. displacement & slip [mm]")
        plt.xlabel("time [day]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, avg_a)
        plt.ylabel("avg. acceleration [m/s2]")
        plt.xlabel("time [day]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, avg_p)
        plt.ylabel("avg. pressure [MPa]")
        plt.xlabel("time [day]")
        plt.grid()
        plt.show()
        return avg_v, avg_z, avg_s, avg_d, avg_a

    def plot_all(self, ts, ys, delta_zero=False):
        N = self.N
        if delta_zero:
            offset = -ys[0, :5 * N]
            ys[:, :5 * N] += offset
        v = ys[:, :N] * self.Vscale
        z = ys[:, N:2 * N] * self.Vscale
        s = ys[:, 2 * N:3 * N] * self.Dscale
        d = ys[:, 3 * N:4 * N] * self.Dscale
        a = ys[:, 4 * N:5 * N] * self.Vscale / (self.Tscale/1000)
        p = ys[:, 5 * N:6 * N] * self.Pscale
        plt.plot(ts * self.Tscale / self.day, v, "k", alpha=.2)
        plt.plot(ts * self.Tscale / self.day, z, "k", alpha=.2)
        plt.ylabel("average velocity and slip rate [m/s]")
        plt.xlabel("time [days]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, d, "k", alpha=.2)
        plt.plot(ts * self.Tscale / self.day, s, "--k", alpha=.2)
        plt.ylabel("avg. displacement & slip [mm]")
        plt.xlabel("time [days]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, a, "k", alpha=.2)
        plt.ylabel("avg. acceleration [m/s2]")
        plt.xlabel("time [day]")
        plt.grid()
        plt.show()
        plt.plot(ts * self.Tscale / self.day, p, "k", alpha=.2)
        plt.ylabel("Pressure [MPa]")
        plt.xlabel("time [days]")
        plt.grid()
        plt.show()
        return v, z, s, d, a, p

    def scale_data(self, xxs):
        vs = xxs[:, :self.N] * self.Vscale
        sr = xxs[:, self.N:2*self.N] * self.Vscale
        ss = xxs[:, 2*self.N:3*self.N] * self.Dscale
        ps = xxs[:, 5*self.N:6*self.N] * self.Pscale

        return vs, sr, ss, ps

    def plot_evolution(self, xxs, ts, delta_zero=False):
        N = self.N
        offset = np.zeros(N * 5)
        if delta_zero:
            offset = -xxs[0, :]
        xxs = offset + xxs
        vs, sr, ss, ps = self.scale_data(xxs)

        vmax = np.amax(vs)
        vmin = np.amin(vs)

        @widgets.interact(incrm=(0, vs.shape[0] - 1, 1), Discretization=False)
        def animate(incrm=0, Discretization=False):
            fig, ax = self.create_fig()
            current_time = ts[incrm]
            print(f'Current time: {current_time}')
            cpl = self.plot_contour(ax=ax,
                                    array=vs,
                                    incr=incrm,
                                    clabel='[m/s]',
                                    cmap='Oranges',
                                    title='slip rate at t=' +
                                    str(round(float(current_time) *
                                        self.Tscale / self.second, 1)) + "s",
                                    show_discretization=Discretization)
            plt.show()

    def create_fig(self, figsize=(6, 5), rows=1, columns=2, w_ratio=[1, .042]):
        fig, ax = plt.subplots(nrows=rows, ncols=columns, gridspec_kw={
                               'width_ratios': w_ratio}, figsize=figsize, constrained_layout=True)
        return fig, ax

    def mesh(self, array, incr, smooth=3.):
        z = np.array([[array[incr, self.nop(i, j)]
                     for i in range(self.Nx)] for j in range(self.Nz)])
        z = scipy.ndimage.zoom(z, smooth)
        x, y = np.meshgrid(np.linspace(0, self.xlength, z.shape[1], endpoint=True),
                           np.linspace(-self.depth_ini, -self.zdepth - self.depth_ini, z.shape[0], endpoint=True))
        return x, y, z

    def nop(self, i, j):
        return i + j * self.Nx

    def colorbar(self, z, ax, clabel="", vmin=None, vmax=None, lvls=10, cmap='Blues', trunc=0.):
        extend, vmin, vmax = self.extend_v(z, vmin, vmax)
        colormap = plt.get_cmap(cmap)
        colormap = self.truncate_colormap(colormap, trunc, 1.)
        bounds = np.linspace(vmin, vmax, lvls)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N, extend=extend)
        cb1 = mpl.colorbar.ColorbarBase(
            ax, cmap=colormap, norm=norm, extend=extend)
        cb1.set_label(clabel, labelpad=10)
        cb1.set_ticks(np.linspace(vmin, vmax, 10))
        return vmin, vmax, colormap, cb1

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def extend_v(self, z, vmin=None, vmax=None):
        extend = "neither"
        if vmax is None:
            vmax = z.max()
        if vmin is None:
            vmin = z.min()
        if vmax < z.max():
            extend = "max"
        if vmin > z.min():
            extend = "min"
        if vmax < z.max() and vmin > z.min():
            extend = "both"
        if vmin == vmax:
            vmax = vmin + 1
        return extend, vmin, vmax

    def contour(self, ax, x, y, z, vmin, vmax, colormap=None, cmap='Blues', lvls=10, title="", cnt=None, show_discretization=False):
        ax.text(0.5, 1.1, title, fontsize=20, horizontalalignment='center',
                verticalalignment='center', label='a', transform=ax.transAxes)
        ax.set_aspect('equal')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [km]')
        if colormap is None:
            colormap = plt.get_cmap(cmap)
        cf = ax.contourf(x, y, z, levels=lvls,
                         cmap=colormap, vmin=vmin, vmax=vmax)
        if cnt is not None:
            for cn in cnt:
                c = ax.contour(x, y, z, levels=cnt, colors="k")
        handles = []
        labels = []
        if show_discretization:
            n, = ax.plot(self.pts_x, -self.pts_z, '+',
                         color='black', alpha=0.4)
            handles.append(n)
            labels.append('Cross Section')
        if cnt is not None:
            for i in range(len(cnt)):
                patch = mpatches.Patch(color='k', linewidth=1)
                handles.append(patch)
                if cmap == 'Blues':
                    labels.append('$\Delta p_f$ = {0} MPa'.format(
                        np.round(cnt[i], 1)))
                else:
                    labels.append('$\Delta \\tau$ = {0} MPa'.format(
                        np.round(cnt[i], 2)))
            leg = ax.legend(handles=handles, labels=labels, loc='upper center')
            patches = leg.get_patches()
            patches[0].set_height(1)
        return cf

    def plot_contour(self, ax, array, incr, trunc=0., smooth=3., clabel="", vmin=None, vmax=None, lvls=10, cmap='Blues', title="", cnt=None, show_discretization=False):
        x, y, z = self.mesh(array, incr, smooth)
        vmin, vmax, colormap, cb1 = self.colorbar(
            z, ax[1], clabel, vmin, vmax, lvls, cmap, trunc)
        cf = self.contour(ax[0], x, y, z, vmin, vmax, colormap,
                          cmap, lvls, title, cnt, show_discretization)
        return cf
